#!/bin/bash
# Generate QA records until a target count, surviving crashes and the laptop.
#
# Runs entirely on the rented box: the generator, the corpus and vLLM are all
# local to each other, so no LLM call crosses the network and nothing depends on
# a laptop staying awake. Each question costs several *sequential* model calls,
# so the round trip that setup removes was the dominant cost, not the compute.
#
#   setsid nohup /workspace/vifin/run_forever.sh easy 60 1500 \
#       > /workspace/vifin/gen_easy.log 2>&1 < /dev/null &
#
# Args: <tier> <batch size> <target records>
set -u

WORK=/workspace/vifin
PY=/workspace/genv/bin/python
TIER="${1:-easy}"
BATCH="${2:-60}"
TARGET="${3:-1500}"

export PYTHONPATH="$WORK/vifinqa-official/src"
export DATA_ROOT="$WORK/data/official_corpus"
export COMPANY_META_PATH="$WORK/data/file_filter.csv"
export QUESTIONS_DIR="$WORK/data/questions"
export CACHE_DIR="$WORK/.cache"
export RUNS_DIR="$WORK/runs"
export OPENAI_URL="http://127.0.0.1:18000/v1"
export OPENAI_API_KEY="local"
cd "$WORK" || exit 1

CONFIG="$WORK/configs/gen_remote_${TIER}.yaml"
POOL="$WORK/runs/${TIER}_pool.jsonl"
PRODUCED="$WORK/runs/remote-${TIER}/per_question.jsonl"
mkdir -p "$WORK/runs"
touch "$POOL"

round=0
dry=0
while true; do
    have=$(wc -l < "$POOL")
    if [[ "$have" -ge "$TARGET" ]]; then
        echo "$(date -u +%H:%M:%S) target reached: $have records"
        break
    fi
    # A fixed seed makes every batch propose the same candidates, so an
    # unattended loop would spend the night regenerating the same questions.
    # The seed is the only thing that has to change between rounds.
    round=$((round + 1))
    sed -i "s/^  seed: .*/  seed: $((round * 7 + 13))/" "$CONFIG"

    echo "$(date -u +%H:%M:%S) round $round: have $have / $TARGET"
    rm -f "$PRODUCED"
    timeout 3600 "$PY" -c "
import sys
sys.argv = ['vifinqa','generate','--config','$CONFIG','--count','$BATCH',
            '--workers','16','--data-root','$DATA_ROOT']
from vifinqa.cli import main
try:
    main()
except SystemExit:
    pass
except Exception as exc:
    print('batch aborted:', type(exc).__name__, str(exc)[:200])
" 2>&1 | tail -4

    added=0
    if [[ -f "$PRODUCED" ]]; then
        added=$(wc -l < "$PRODUCED")
        cat "$PRODUCED" >> "$POOL"
        rm -f "$PRODUCED"
    fi
    echo "$(date -u +%H:%M:%S) round $round added $added"

    # A round that yields nothing twice running means the candidate pool for
    # this tier is exhausted or the server is unwell; stop rather than burn
    # rented hours producing nothing.
    if [[ "$added" -eq 0 ]]; then
        dry=$((dry + 1))
        if [[ "$dry" -ge 3 ]]; then
            echo "$(date -u +%H:%M:%S) three empty rounds, stopping"
            break
        fi
    else
        dry=0
    fi
    sleep 3
done
echo "$(date -u +%H:%M:%S) finished with $(wc -l < "$POOL") records in $POOL"
