#!/bin/bash
# Wait up to ~25 minutes for vLLM
for i in $(seq 1 100); do
  if curl -sf http://127.0.0.1:18000/v1/models >/tmp/m.json; then
    echo READY
    cat /tmp/m.json
    exit 0
  fi
  if (( i % 5 == 0 )); then
    echo "wait_$i size=$(du -sh /workspace/hf/hub/models--Qwen--Qwen3-14B 2>/dev/null | cut -f1) incomplete=$(find /workspace/hf/hub/models--Qwen--Qwen3-14B -name '*.incomplete' 2>/dev/null | wc -l)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ' -
  fi
  sleep 15
done
echo FAIL
tail -30 /workspace/logs/vllm.log
exit 1
