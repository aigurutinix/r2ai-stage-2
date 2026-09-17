#!/bin/bash
# Set the rented box up to generate on its own, with nothing on the laptop.
#
# Until now the generator ran locally and reached the model through an SSH
# tunnel, which makes the run hostage to a laptop lid and sends every LLM call
# from Vietnam to Finland and back. Each question costs several *sequential*
# calls, so that round trip was buying latency, not compute. Moving the
# generator onto the box removes both problems at once: it survives the laptop
# and it talks to vLLM over localhost.
#
# Run:  bash remote_bootstrap.sh
set -euo pipefail

WORK=/workspace/vifin
mkdir -p "$WORK"
cd "$WORK"

echo "== unpacking =="
for bundle in corpus_bundle.tar.gz code_bundle.tar.gz; do
    if [[ -f "/workspace/$bundle" ]]; then
        tar -xzf "/workspace/$bundle" -C "$WORK"
        echo "  $bundle"
    fi
done

echo "== python env =="
source /venv/main/bin/activate
# The image already carries torch and vllm; only the generator's own
# dependencies are missing. `underthesea` pulls a large wheel set, so this is
# the slow step and it is worth letting finish before launching anything.
uv pip install -q bm25s underthesea sentence-transformers pydantic-settings \
    tenacity typer pyyaml python-dotenv openai 2>&1 | tail -3 || true

echo "== sanity =="
export PYTHONPATH="$WORK/vifinqa-official/src"
export DATA_ROOT="$WORK/data/official_corpus"
export COMPANY_META_PATH="$WORK/data/file_filter.csv"
export QUESTIONS_DIR="$WORK/data/questions"
export CACHE_DIR="$WORK/.cache"
export RUNS_DIR="$WORK/runs"
export OPENAI_URL="http://127.0.0.1:18000/v1"
export OPENAI_API_KEY="local"

python -c "
import sys
sys.argv = ['vifinqa', 'catalog', '--data-root', '$DATA_ROOT']
from vifinqa.cli import main
try:
    main()
except SystemExit:
    pass
"

cat > "$WORK/run_forever.sh" <<'INNER'
#!/bin/bash
# Keep generating until the target is reached, restarting after any crash.
# A generation run that dies at hour three has cost three hours of a rented
# card; the loop is what makes an unattended overnight run worth starting.
set -u
WORK=/workspace/vifin
source /venv/main/bin/activate
export PYTHONPATH="$WORK/vifinqa-official/src"
export DATA_ROOT="$WORK/data/official_corpus"
export COMPANY_META_PATH="$WORK/data/file_filter.csv"
export QUESTIONS_DIR="$WORK/data/questions"
export CACHE_DIR="$WORK/.cache"
export RUNS_DIR="$WORK/runs"
export OPENAI_URL="http://127.0.0.1:18000/v1"
export OPENAI_API_KEY="local"
cd "$WORK"

TIER="${1:-easy}"
BATCH="${2:-60}"
TARGET="${3:-1500}"
OUT="$WORK/runs/${TIER}_pool.jsonl"
mkdir -p "$WORK/runs"
touch "$OUT"

while true; do
    have=$(wc -l < "$OUT")
    if [[ "$have" -ge "$TARGET" ]]; then
        echo "$(date -u +%H:%M:%S) target reached: $have"
        break
    fi
    echo "$(date -u +%H:%M:%S) have $have / $TARGET, generating $BATCH more"
    # Each batch writes to its own run directory, then is appended to the pool.
    # vLLM staying up across batches is the point: model load is minutes.
    stamp=$(date -u +%H%M%S)
    python -c "
import sys
sys.argv = ['vifinqa','generate','--config','$WORK/configs/gen_remote_${TIER}.yaml',
            '--count','$BATCH','--workers','16','--data-root','$DATA_ROOT']
from vifinqa.cli import main
try:
    main()
except SystemExit:
    pass
except Exception as exc:
    print('batch failed:', type(exc).__name__, exc)
" 2>&1 | tail -3
    produced="$WORK/runs/remote-${TIER}/per_question.jsonl"
    if [[ -f "$produced" ]]; then
        cat "$produced" >> "$OUT"
        rm -f "$produced"
    fi
    sleep 5
done
INNER
chmod +x "$WORK/run_forever.sh"

echo
echo "ready. start an unattended run with:"
echo "  setsid nohup $WORK/run_forever.sh easy 60 1500 > $WORK/gen.log 2>&1 < /dev/null &"
echo "check on it with:"
echo "  wc -l $WORK/runs/easy_pool.jsonl ; tail -5 $WORK/gen.log"
