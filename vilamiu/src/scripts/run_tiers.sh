#!/bin/bash
# Generate across all three tiers, in the proportion the training actually needs.
#
# The `easy` tier produces single-cell lookups â€” exactly what our lexical label
# matcher already answers at 42.8%. Training only on those teaches the half we
# win and risks biasing the model toward one-cell answers on the half we lose.
# The derived tiers are the ones that matter, and they are the slower ones, so
# they get the larger share of the night.
#
#   setsid nohup /workspace/vifin/run_tiers.sh > /workspace/vifin/gen_all.log 2>&1 < /dev/null &
set -u

WORK=/workspace/vifin
PY=/workspace/genv/bin/python

export PYTHONPATH="$WORK/vifinqa-official/src"
export DATA_ROOT="$WORK/data/official_corpus"
export COMPANY_META_PATH="$WORK/data/file_filter.csv"
export QUESTIONS_DIR="$WORK/data/questions"
export CACHE_DIR="$WORK/.cache"
export RUNS_DIR="$WORK/runs"
export OPENAI_URL="http://127.0.0.1:18000/v1"
export OPENAI_API_KEY="local"
# The embedder shares the card with vLLM rather than being masked off it.
# Keeping it on CPU was the safe choice and it cost a factor of sixty: the medium
# tier produced 1 record in 64 minutes with 12 cores pinned at 1195% and vLLM
# reporting 0 tokens/s — the GPU idle while the CPU did work it is bad at.
# vLLM is now capped at 0.70 utilisation, which leaves ~7 GB against the ~2.5 GB
# bge-m3 needs.
cd "$WORK" || exit 1

run_tier () {
    local tier="$1" batch="$2" target="$3"
    local config="$WORK/configs/gen_remote_${tier}.yaml"
    local pool="$WORK/runs/${tier}_pool.jsonl"
    local produced="$WORK/runs/remote-${tier}/per_question.jsonl"
    [[ -f "$config" ]] || { echo "no config for $tier, skipping"; return; }
    mkdir -p "$WORK/runs"
    touch "$pool"

    local round=0 dry=0
    while true; do
        local have
        have=$(wc -l < "$pool")
        if [[ "$have" -ge "$target" ]]; then
            echo "$(date -u +%H:%M:%S) [$tier] target reached: $have"
            return
        fi
        round=$((round + 1))
        # The seed shuffles the candidate list, so it is the only thing that
        # varies what a batch proposes. Deriving it from the round number looks
        # fine until the loop is restarted: round 1 recomputes the first seed and
        # regenerates the first batch verbatim â€” 7 of 9 records in the second run
        # were byte-identical to the first. Seed from the clock and the pool size,
        # neither of which resets.
        sed -i "s/^  seed: .*/  seed: $(( ($(date +%s) + have * 31) % 999983 ))/" "$config"
        echo "$(date -u +%H:%M:%S) [$tier] round $round: have $have / $target"
        rm -f "$produced"
        timeout 5400 "$PY" -c "
import sys
sys.argv = ['vifinqa','generate','--config','$config','--count','$batch',
            '--data-root','$DATA_ROOT']
from vifinqa.cli import main
try:
    main()
except SystemExit:
    pass
except Exception as exc:
    print('batch aborted:', type(exc).__name__, str(exc)[:200])
" 2>&1 | tail -4

        # Even with a fresh seed, `is_table_eligible` rejects most tables, so the
        # surviving pool is small and popular tables recur across rounds. Append
        # only what the pool has not already got, keyed on the table cited and the
        # question text â€” a duplicate costs GPU hours and, worse, over-weights one
        # table in the training mix.
        local added=0
        if [[ -f "$produced" ]]; then
            local before
            before=$(wc -l < "$pool")
            "$PY" - "$produced" "$pool" <<'DEDUP'
import json, sys
produced, pool = sys.argv[1], sys.argv[2]
def key(rec):
    return (rec.get("question", "").strip(),
            tuple(rec.get("relevant_tables") or []))
seen = set()
try:
    with open(pool, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                seen.add(key(json.loads(line.replace(": NaN", ": null"))))
except FileNotFoundError:
    pass
kept = 0
with open(pool, "a", encoding="utf-8") as out, open(produced, encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        rec = json.loads(line.replace(": NaN", ": null"))
        k = key(rec)
        if k in seen:
            continue
        seen.add(k)
        out.write(line if line.endswith("\n") else line + "\n")
        kept += 1
print(f"  deduped: kept {kept}", flush=True)
DEDUP
            local after
            after=$(wc -l < "$pool")
            added=$((after - before))
            rm -f "$produced"
        fi
        echo "$(date -u +%H:%M:%S) [$tier] round $round added $added (pool now $(wc -l < "$pool"))"
        if [[ "$added" -eq 0 ]]; then
            dry=$((dry + 1))
            [[ "$dry" -ge 3 ]] && { echo "$(date -u +%H:%M:%S) [$tier] three empty rounds, moving on"; return; }
        else
            dry=0
        fi
        sleep 3
    done
}

# easy is capped low on purpose: it is the tier we least need.
run_tier easy 60 400
run_tier medium 40 500
run_tier intermediate 40 500
echo "$(date -u +%H:%M:%S) ALL DONE"
for t in easy medium intermediate; do
    f="$WORK/runs/${t}_pool.jsonl"
    [[ -f "$f" ]] && echo "  $t: $(wc -l < "$f") records"
done
