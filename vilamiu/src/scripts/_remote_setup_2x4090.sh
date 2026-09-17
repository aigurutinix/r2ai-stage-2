#!/bin/bash
# Bootstrap 2x4090: install vLLM, serve Qwen3-14B fp16 TP=2 on :18000
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export NVIDIA_VISIBLE_DEVICES=0,1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/hf
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p /workspace/vifin /workspace/hf /workspace/logs

# Prefer the instance venv if present.
if [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

NEED_INSTALL=0
python - <<'PY' || NEED_INSTALL=1
import importlib.util, sys
need = [m for m in ("torch", "vllm", "openai")
        if importlib.util.find_spec(m) is None]
print("missing:", need)
sys.exit(1 if need else 0)
PY

if [ "$NEED_INSTALL" -ne 0 ]; then
  echo "Installing torch + vllm (this takes a few minutes)..."
  pip install -U pip wheel
  # CUDA 12.8 image — use matching torch/vllm wheels from default index.
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install vllm openai
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "gpus", torch.cuda.device_count())
assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
import vllm
print("vllm", vllm.__version__)
PY

# Kill any previous serve.
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

nohup vllm serve Qwen/Qwen3-14B \
  --host 0.0.0.0 \
  --port 18000 \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  > /workspace/logs/vllm.log 2>&1 &

echo "vllm_pid=$!"
echo "tailing log until ready..."
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:18000/v1/models >/tmp/models.json; then
    echo READY
    cat /tmp/models.json
    exit 0
  fi
  echo "wait_$i mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
  sleep 20
done
echo FAIL
tail -80 /workspace/logs/vllm.log
exit 1
