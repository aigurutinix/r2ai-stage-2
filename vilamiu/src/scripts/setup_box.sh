#!/bin/bash
# Bring a rented vast.ai vLLM-template box to a state where train_qlora.py runs.
#
# Every step here is something that went wrong on a real box, not a precaution:
#
#   * The template auto-starts a vLLM server on Qwen3.5-9B. `supervisorctl stop
#     vllm` stops the supervisor wrapper and leaves `VLLM::EngineCore` alive
#     holding 19 GB — nvidia-smi then reports the holder as `[Not Found]`,
#     because it is our own process seen from the host PID namespace, which
#     reads exactly like another tenant squatting on the card. Kill by PID.
#   * `pkill -f vllm` is not the answer: the remote command string contains the
#     pattern, so pkill matches its own invocation and killed the ssh session
#     four times.
#   * The template has already downloaded ~15 GB of Qwen3.5-9B into
#     /workspace/models before you log in. On a small disk that is the difference
#     between fitting the base model and not.
#   * `pip install` without --no-deps moved torch twice on earlier boxes. The
#     symptom, `operator torchvision::nms does not exist`, appeared hours later
#     with nothing in the traceback pointing at the install.
#   * --no-deps then leaves leaf packages missing. The ones actually needed are
#     discovered below by importing until it stops failing, rather than guessed.
#
# Usage:  bash scripts/setup_box.sh
set -uo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader

echo
echo "== stop the template's model server =="
supervisorctl stop vllm model-ui ray 2>/dev/null | sed 's/^/  /'
# The wrapper exits; the engine does not. Match on the process image, and take
# the PID from ps rather than letting pkill pattern-match this script.
for pid in $(ps -eo pid,cmd --no-headers | grep -E "VLLM::EngineCore|vllm serve" | grep -v grep | awk '{print $1}'); do
  echo "  killing orphan $pid"
  kill -9 "$pid" 2>/dev/null
done
sleep 10
echo "  VRAM now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

echo
echo "== reclaim the pre-downloaded model =="
du -sh /workspace/models 2>/dev/null | sed 's/^/  was: /'
rm -rf /workspace/models/* 2>/dev/null
df -h / | tail -1 | sed 's/^/  /'

echo
echo "== python =="
source /venv/main/bin/activate
python -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

echo
echo "== training packages, pinned away from the resolver =="
pip install -q --no-deps peft trl datasets multiprocess dill xxhash pandas pyarrow 2>&1 | tail -2
# --no-deps means the leaf set is incomplete by construction. Import, install
# whatever the ImportError names, repeat. Six rounds is far more than the two
# this has ever needed.
for _ in 1 2 3 4 5 6; do
  missing=$(python -c "
try:
    import datasets, trl, peft, transformers, bitsandbytes, accelerate
    print('OK')
except ModuleNotFoundError as exc:
    print(exc.name)
" 2>/dev/null)
  if [ "$missing" = "OK" ]; then break; fi
  echo "  missing $missing"
  pip install -q --no-deps "$missing" 2>&1 | tail -1
done

echo
echo "== versions =="
python - <<'PY'
import importlib, torch
print("  torch", torch.__version__, "cuda available", torch.cuda.is_available())
assert torch.cuda.is_available(), "the install moved torch — reinstall it pinned"
for name in ("transformers", "peft", "trl", "datasets", "bitsandbytes", "accelerate"):
    print("  %-14s %s" % (name, getattr(importlib.import_module(name), "__version__", "?")))
PY

echo
cat <<'NEXT'
== ready ==

Upload, from the laptop:

  scp -P <port> scripts/train_qlora.py root@<host>:/workspace/vifin/scripts/
  scp -P <port> artifacts/sft_locate.jsonl artifacts/sft_prog.jsonl \
      root@<host>:/workspace/vifin/artifacts/

Then train. On a 100 GB disk the official bf16 weights (29.5 GB) fit; on a 32 GB
disk use unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit (9.9 GB), which is the same
model already stored in the NF4 format train_qlora.py would quantise it to. The
script detects a pre-quantised checkpoint and skips its own BitsAndBytesConfig.

  cd /workspace/vifin && source /venv/main/bin/activate
  export HF_HUB_ENABLE_HF_TRANSFER=1
  nohup python scripts/train_qlora.py \
      --data artifacts/sft_locate.jsonl \
      --out /workspace/lora_locate \
      --base Qwen/Qwen2.5-Coder-14B-Instruct \
      --epochs 2 --max-len 12288 --batch 1 --accum 16 --rank 32 \
      > /workspace/train_locate.log 2>&1 &

Two lines decide whether the run is real. Both were absent when an earlier run
trained for hours, saved cleanly, and produced an adapter identical to the base
on all 57 evaluated questions:

  LoRA layers attached: N   trainable params: M    <- measured 336 / 34,406,400
  lora_B tensors that moved off zero: N            <- measured 336

Do not lower --max-len to save VRAM. Pairs over the cap are dropped, not
truncated: at 4096 only 5 of 40 survived, while at 12288 none of 22 were lost
(median 6,942 tokens, longest 11,674). A run that silently keeps a tenth of its
data still reports a healthy loss.

Measured rate: 6.6 s/pair, so 1,562 pairs x 2 epochs is about 5.7 hours.
Checkpoints are written each epoch, so epoch 1 is usable if you stop early.
NEXT
