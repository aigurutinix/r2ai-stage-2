#!/usr/bin/env bash
# Bring the whole demo up against a rented Vast.ai box, from one pasted SSH line.
#
#     demo/vast_up.sh 'ssh -p 12345 root@1.2.3.4'
#
# Copy that line straight from the Vast instance card. The script installs vLLM
# on the box if it is missing, starts the planner and reranker, tunnels both
# ports back here, starts the local UI, and asks one real question to prove the
# whole path works. Stop everything again with demo/vast_down.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

# Left empty, the box picks its own checkpoints from the card it has.
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-}"
RERANK_CHECKPOINT="${RERANK_CHECKPOINT:-}"
# Served under the id the pipeline requests, so no client-side override is needed.
PLANNER_NAME="${PLANNER_NAME:-Qwen/Qwen3.5-9B}"
RERANK_NAME="${RERANK_NAME:-Qwen3-Reranker-8B}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
DEMO_PORT="${DEMO_PORT:-8000}"
STATE_DIR="${STATE_DIR:-demo/.demo-state}"
SOCKET="$STATE_DIR/ssh.sock"

mkdir -p "$STATE_DIR"

[[ $# -ge 1 ]] || { echo "usage: $0 'ssh -p PORT USER@HOST'" >&2; exit 2; }

# Vast's card gives a whole command line, sometimes with its own -L forwards.
# Keep the host, the port and any identity file; drop everything else.
TARGET="" PORT=22 IDENTITY=()
SSH_LINE="$1"
args=($SSH_LINE)
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    ssh) ;;
    -p) PORT="${args[++i]}" ;;
    -i) IDENTITY+=(-i "${args[++i]/#\~/$HOME}") ;;
    -L|-R|-D) ((i++)) ;;
    -*) ;;
    *) [[ -z "$TARGET" ]] && TARGET="${args[i]}" ;;
  esac
done
[[ -n "$TARGET" ]] || { echo "no user@host found in: $SSH_LINE" >&2; exit 2; }
# Vast's own line never carries -i, but its key is usually the one it installed.
if [[ ${#IDENTITY[@]} -eq 0 && -f "$HOME/.ssh/vastai" ]]; then
  IDENTITY=(-i "$HOME/.ssh/vastai")
fi

SSH=(ssh -p "$PORT" "${IDENTITY[@]}"
     -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30
     -o ConnectTimeout=15)

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Opening a reusable connection to $TARGET"
# A master from an earlier run keeps its forwarded ports even after its socket
# file is deleted, so shut it down properly and free 8001/8002 before opening a
# new one. Otherwise every rerun leaks a connection and the forward fails.
[[ -S "$SOCKET" ]] && ssh -S "$SOCKET" -O exit "$TARGET" 2>/dev/null || true
rm -f "$SOCKET"
fuser -k -n tcp 8001 8002 2>/dev/null || true
"${SSH[@]}" -M -S "$SOCKET" -fN "$TARGET"
on_box() { ssh -S "$SOCKET" "$TARGET" "$@"; }

say "Preparing the box (installs vLLM $VLLM_VERSION only if missing)"
on_box "PLANNER_OVERRIDE='$PLANNER_CHECKPOINT' RERANK_OVERRIDE='$RERANK_CHECKPOINT' bash -s" <<REMOTE
set -uo pipefail
set -e
mkdir -p ~/vifin-demo
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate
command -v vllm >/dev/null || pip install --quiet "vllm==$VLLM_VERSION"

# The Vast vLLM image boots its own single-model server, which holds most of the
# card. This demo needs two servers of its own, so stand that one down.
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl stop vllm >/dev/null 2>&1 || true
  echo "stopped the image's bundled vllm service"
fi
if curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 \
   && curl -sf -m 3 http://127.0.0.1:8002/health >/dev/null 2>&1; then
  echo "both servers already serving, leaving them up"
  DRAIN=no
else
  # vLLM's engine core renames itself, so matching only "vllm serve" leaves the
  # process that actually holds the card alive and the drain never completes.
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  DRAIN=yes
fi

# Releasing an 18 GB model takes longer than its process takes to exit, and a
# server starting too early sizes its cache against memory still held.
echo "waiting for the card to drain"
used=0
[ "\$DRAIN" = no ] && used=0
for _ in \$(seq 1 30); do
  [ "\$DRAIN" = no ] && break
  used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "\$used" -lt 1000 ] && break
  sleep 2
done
echo "card has \$used MiB in use"

serve() {  # name port model served extra...
  local name=\$1 port=\$2 model=\$3 served=\$4; shift 4
  if curl -sf "http://127.0.0.1:\$port/health" >/dev/null 2>&1; then
    echo "\$name already serving on \$port"; return
  fi
  echo "starting \$name (\$model) on \$port"
  setsid nohup vllm serve "\$model" --host 127.0.0.1 --port "\$port" \
    --served-model-name "\$served" "\$@" \
    < /dev/null > ~/vifin-demo/\$name.log 2>&1 &
  echo "waiting for \$name (a first run downloads weights, allow ~15 min)"
  for _ in \$(seq 1 180); do
    curl -sf "http://127.0.0.1:\$port/health" >/dev/null 2>&1 && { echo "\$name ready"; return; }
    sleep 10
  done
  echo "\$name never came up. From ~/vifin-demo/\$name.log:" >&2
  grep -E "Error|ValueError|RuntimeError" ~/vifin-demo/\$name.log | tail -5 >&2
  exit 1
}

# FP8 needs Ada or newer. On an Ampere card such as a 3090 there are no FP8
# tensor cores, so fall back to AWQ instead.
CAP=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
VRAM=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "\$GPU, compute \$CAP, \$VRAM MiB"

# FP8 needs Ada or newer. On Ampere there are no FP8 tensor cores, so the
# planner falls back to AWQ.
if [ "\${CAP:-0}" -ge 89 ]; then
  DEFAULT_PLANNER=lovedheart/Qwen3.5-9B-FP8
else
  DEFAULT_PLANNER=QuantTrio/Qwen3.5-9B-AWQ
fi

# The reranker is chosen by what is left after the planner, not by compute
# capability. The planner needs about 13 GiB; the 8B reranker needs about 18 and
# only fits beside it on a 40 GiB card or larger.
if [ "\${VRAM:-0}" -ge 40000 ]; then
  DEFAULT_RERANK=Qwen/Qwen3-Reranker-8B
  PLANNER_EXTRA="--gpu-memory-utilization 0.34"
  RERANK_EXTRA="--gpu-memory-utilization 0.46"
else
  # The planner alone needs about 15 GiB before it has a single cache block, so
  # a 24 GiB card leaves under 5 for the reranker. Only the 4-bit 4B fits there.
  echo "under 40 GiB: using the quantised 4B reranker so both models fit"
  DEFAULT_RERANK=datapizza-ai-lab/Qwen3-Reranker-4B-AWQ-4bit
  PLANNER_EXTRA="--gpu-memory-utilization 0.65"
  RERANK_EXTRA="--gpu-memory-utilization 0.20"
fi
PLANNER=\${PLANNER_OVERRIDE:-\$DEFAULT_PLANNER}
RERANK=\${RERANK_OVERRIDE:-\$DEFAULT_RERANK}

# One at a time. vLLM sizes its KV cache against free memory at startup, so two
# servers booting together both measure the same free bytes and overcommit.
# Qwen3.5 is a hybrid Mamba model: every concurrent sequence claims its own
# Mamba cache block, and the default 256 needs far more of them than a 24 GB
# card has. The demo asks one question at a time, so 16 is generous.
serve planner 8001 "\$PLANNER" "$PLANNER_NAME" \$PLANNER_EXTRA --max-model-len 8192 --max-num-seqs 16
# Served as an ordinary generative model. The pipeline scores a pair itself by
# reading the "yes" and "no" logits under the instruction the frozen run used;
# vLLM's own rerank route substitutes a generic web-search instruction and
# offers no way to pass ours, which on one check dropped the gold table from
# rank 1 to rank 21 of 50.
serve reranker 8002 "\$RERANK" "$RERANK_NAME" \$RERANK_EXTRA \
  --max-model-len 8192 --max-num-seqs 32
REMOTE

say "Tunnelling 8001 and 8002 to this machine"
ssh -S "$SOCKET" -O forward -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 "$TARGET"

# The corpus lives outside this repo in the usual checkout, and the pipeline
# refuses to start without it.
if [[ -z "${VIFINQA_DATA_ROOT:-}" ]]; then
  for candidate in data/raw/vifinqa ../data/raw/vifinqa; do
    [[ -d "$candidate" ]] && { VIFINQA_DATA_ROOT="$candidate"; break; }
  done
fi
[[ -n "${VIFINQA_DATA_ROOT:-}" && -d "$VIFINQA_DATA_ROOT" ]] || {
  echo "corpus not found. Set VIFINQA_DATA_ROOT to the vifinqa data directory." >&2
  exit 1; }
say "Using corpus at $VIFINQA_DATA_ROOT"

# The live path imports these lazily, so a missing one only shows up as a
# failed question rather than a failed startup.
PY_BIN="${PYTHON:-python3}"
if ! "$PY_BIN" -c "import openai, pydantic" 2>/dev/null; then
  echo "installing the demo extras into $PY_BIN"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$PY_BIN" openai pydantic
  else
    "$PY_BIN" -m pip install --quiet openai pydantic
  fi
fi

say "Starting the local UI on port $DEMO_PORT"
fuser -k -n tcp "$DEMO_PORT" 2>/dev/null || true
DEMO_MODE=remote \
PLANNER_URL=http://127.0.0.1:8001 \
RERANK_URL=http://127.0.0.1:8002 \
RERANKER_MODEL="$RERANK_NAME" \
DEMO_PORT="$DEMO_PORT" \
VIFINQA_DATA_ROOT="$VIFINQA_DATA_ROOT" \
  nohup demo/launch_demo.sh > "$STATE_DIR/demo.log" 2>&1 &
echo $! > "$STATE_DIR/demo.pid"
printf '%s\n%s\n' "$TARGET" "$PORT" > "$STATE_DIR/target"

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$DEMO_PORT/api/health" >/dev/null 2>&1 && break
  sleep 2
done

say "Checking the connection"
health=$(curl -sf "http://127.0.0.1:$DEMO_PORT/api/health") || {
  echo "the local UI never answered. Log:" >&2; tail -20 "$STATE_DIR/demo.log" >&2; exit 1; }
echo "$health"
grep -q '"ready": *true' <<<"$health" || {
  echo "the pipeline reports it is not ready; see the endpoints field above" >&2; exit 1; }

say "Asking one real question end to end"
if demo/warmup.sh | tee "$STATE_DIR/warmup.log" | grep -q '"type": *"result"'; then
  say "All good. Open http://127.0.0.1:$DEMO_PORT"
  echo "Stop everything with: demo/vast_down.sh"
else
  echo "the question did not produce an answer; full stream in $STATE_DIR/warmup.log" >&2
  exit 1
fi
