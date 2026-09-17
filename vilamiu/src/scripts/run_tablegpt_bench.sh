#!/bin/bash
# Serve each TableGPT model in turn and score it on the same 100 questions the
# five OpenRouter models were scored on, so the numbers land in one table.
#
# Usage on the box:  bash /workspace/run_tablegpt_bench.sh
#
# Design notes worth keeping:
#
# * The card is reclaimed by PID from nvidia-smi before each serve. `kill $PID`
#   on the API server leaves its EngineCore child holding all 24 GB, which is
#   what OOMed two runs earlier today, and `pkill -f` matches its own invocation.
# * Inputs are checked before a model is downloaded, because the first attempt at
#   this shape died four times in a row on one missing prompt file.
# * Weights come down once per model and are kept, so a re-run is cheap.
set -uo pipefail
cd /workspace/vifin
source /venv/main/bin/activate
export PYTHONPATH=src HF_HOME=/workspace/.hf_home HF_HUB_ENABLE_HF_TRANSFER=1
exec >>/workspace/tablegpt.log 2>&1
echo "=== start $(date -u)"

for f in vifinqa-official/prompts/answering/program_system.txt \
         artifacts/tables.parquet artifacts/easy_full.jsonl \
         artifacts/_gold_anchor_rank.jsonl; do
    [ -s "$f" ] || { echo "MISSING: $f"; exit 1; }
done
python -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from vifin.answering.generate import build_prompts
s,u = build_prompts(Path('.'))
print(f'prompts ok: {len(s)} + {len(u)} chars')
" || exit 1

reclaim() {
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -9 "$pid" 2>/dev/null
    done
    for _ in $(seq 1 30); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        [ "${used:-9999}" -lt 1000 ] && break
        sleep 2
    done
}

bench() {  # $1 = model id, $2 = short name
    local model="$1" name="$2"
    echo ""
    echo "########## $name  ($model)  $(date -u)"
    reclaim
    vllm serve "$model" --max-model-len 16384 --gpu-memory-utilization 0.88 \
        --host 127.0.0.1 --port 18000 > "/workspace/vllm_$name.log" 2>&1 &
    local pid=$!
    local ready=0
    for i in $(seq 1 240); do   # weights download on the first pass
        curl -s http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q "$model" && {
            ready=1; echo "  server ready after ${i}0s"; break; }
        kill -0 $pid 2>/dev/null || { echo "  server died"; break; }
        sleep 10
    done
    if [ "$ready" != 1 ]; then
        echo "  FAILED to serve $model"
        grep -iE "error|not supported|OutOfMemory" "/workspace/vllm_$name.log" | tail -5 | cut -c1-200
        kill -9 $pid 2>/dev/null
        return 1
    fi
    python scripts/_probe_self_consistency.py --limit 100 --samples 1 \
        --workers 6 --temperature 0.0 --model "$model" \
        --local-url http://127.0.0.1:18000/v1 \
        --out "artifacts/_sc_$name.jsonl"
    kill -9 $pid 2>/dev/null
    reclaim
}

bench "tablegpt/TableGPT2-7B" tablegpt2
bench "tablegpt/TableGPT-R1"  tablegptr1

echo ""
echo "=== done $(date -u)"
echo "compare against, on the same 100 questions:"
echo "   qwen3-14b ~38%   ministral-14b 24.0%   granite-4.1-8b 20.0%"
echo "   gemma-3-12b 20.0%   phi-4 19.0%"
