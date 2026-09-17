#!/bin/bash
# Wait for training and evaluation to finish, copy everything off the rented box,
# verify it locally, and only then destroy the instance.
#
# The destroy is irreversible and takes the disk with it. Earlier today a
# finished 5.5-hour run was lost exactly this way — the machine went away while
# its adapter existed nowhere else. So nothing here trusts the box's own report
# that a file was written: every artefact is fetched to this machine and opened
# here before the instance is touched.
#
# Verification, all of which must pass:
#   1. adapter_model.safetensors present locally, over 200 MB, and readable as
#      safetensors — a truncated copy has the right name and the wrong contents
#   2. adapter_config.json parses, names the Qwen2.5-Coder-14B base, rank 32
#   3. evalgen_final.jsonl has at least 70 of its 78 rows
#   4. eval.log contains the completion marker
#
# Any failure leaves the instance running and says why. Billing for an idle box
# is $0.34/hour; re-renting and re-training is $1 and 2.7 hours.
#
# Usage: bash scripts/finish_and_destroy.sh <port> <host> <instance_id>
set -uo pipefail

PORT="${1:?usage: finish_and_destroy.sh <port> <host> <instance_id>}"
HOST="${2:?usage: finish_and_destroy.sh <port> <host> <instance_id>}"
INSTANCE="${3:?usage: finish_and_destroy.sh <port> <host> <instance_id>}"
LOCAL="${LOCAL_DIR:-artifacts/lora_pulled}"
O=(-o StrictHostKeyChecking=no -o ConnectTimeout=30)

mkdir -p "$LOCAL"
say() { echo "$(date +%H:%M:%S) $*"; }

# Completion is detected by a marker the training script writes, not by the
# absence of a process. `pgrep -f <pattern>` run over ssh matches its own
# invocation, because the remote command string contains the pattern — so the
# wait never ends. This is the third appearance of that shape in this project:
# `pkill -f` killed the ssh session four times, and a monitor reported a finished
# run as still training for half an hour. A log marker cannot match itself.
say "waiting for training to finish"
while ! ssh -p "$PORT" "${O[@]}" "root@$HOST" \
        'grep -q "adapter written to" /workspace/train_locate.log 2>/dev/null' 2>/dev/null; do
  sleep 300
done
say "training finished (adapter written)"

say "waiting for evaluation to finish"
for _ in $(seq 1 96); do
  if ssh -p "$PORT" "${O[@]}" "root@$HOST" \
        'grep -q "EVAL DONE" /workspace/eval.log 2>/dev/null' 2>/dev/null; then
    say "evaluation finished"
    break
  fi
  sleep 300
done

say "fetching artefacts"
scp -q -P "$PORT" "${O[@]}" \
    "root@$HOST:/workspace/lora_locate/adapter_model.safetensors" \
    "root@$HOST:/workspace/lora_locate/adapter_config.json" \
    "$LOCAL/" 2>/dev/null
scp -q -P "$PORT" "${O[@]}" \
    "root@$HOST:/workspace/evalgen_final.jsonl" \
    "root@$HOST:/workspace/eval.log" \
    "root@$HOST:/workspace/train_locate.log" \
    "$LOCAL/" 2>/dev/null

say "verifying locally"
python - "$LOCAL" <<'PY'
import json, sys
from pathlib import Path

local = Path(sys.argv[1])
problems = []

adapter = local / "adapter_model.safetensors"
if not adapter.exists():
    problems.append("adapter_model.safetensors missing")
elif adapter.stat().st_size < 200_000_000:
    problems.append(f"adapter only {adapter.stat().st_size/1e6:.0f} MB — truncated")
else:
    try:
        from safetensors import safe_open
        with safe_open(str(adapter), framework="pt") as handle:
            keys = list(handle.keys())
        if not keys:
            problems.append("adapter has no tensors")
        else:
            print(f"  adapter: {adapter.stat().st_size/1e6:.0f} MB, {len(keys)} tensors")
    except Exception as exc:
        problems.append(f"adapter unreadable: {type(exc).__name__}")

config = local / "adapter_config.json"
if not config.exists():
    problems.append("adapter_config.json missing")
else:
    try:
        cfg = json.loads(config.read_text(encoding="utf-8"))
        base = str(cfg.get("base_model_name_or_path", ""))
        if "Qwen2.5-Coder-14B" not in base:
            problems.append(f"unexpected base {base!r}")
        if int(cfg.get("r", 0)) != 32:
            problems.append(f"unexpected rank {cfg.get('r')!r}")
        print(f"  config: base {base}, r={cfg.get('r')}, alpha={cfg.get('lora_alpha')}")
    except Exception as exc:
        problems.append(f"config unreadable: {type(exc).__name__}")

gen = local / "evalgen_final.jsonl"
rows = 0
if gen.exists():
    rows = sum(1 for line in gen.read_text(encoding="utf-8").splitlines() if line.strip())
if rows < 70:
    problems.append(f"evalgen_final.jsonl has {rows} rows, expected ~78")
else:
    print(f"  evaluation: {rows} rows")

log = local / "eval.log"
if not log.exists() or "EVAL DONE" not in log.read_text(encoding="utf-8", errors="replace"):
    problems.append("eval.log missing its completion marker")

if problems:
    print("\nNOT SAFE TO DESTROY:")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)
print("\nall checks passed")
PY

if [ $? -ne 0 ]; then
  say "verification FAILED — instance left running, nothing destroyed"
  exit 1
fi

say "destroying instance $INSTANCE"
ssh -p "$PORT" "${O[@]}" "root@$HOST" \
    "/opt/instance-tools/bin/vastai destroy instance $INSTANCE" 2>&1 | tail -3
say "done"
