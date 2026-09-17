#!/bin/bash
# Generate paper-engine records across many seeds in parallel.
#
# Two facts about `paper_pipeline.py` shape this, both found by reading the code
# and then confirming in the output:
#
#   * `_scenario(pool)` is called ONCE, outside `for _ in range(count)`, so every
#     record a process writes comes from the same twelve candidate tables. A run
#     with `count: 500` yields 500 records about a handful of tables — three of
#     five sampled records were the same question about CEO's 2017 EPS, two of
#     them word for word. Diversity comes from seeds, not from count.
#   * The loop is serial, but nothing is shared between processes, so N seeds run
#     N-ways parallel with no code change. Measured: 4 processes finished 45 calls
#     in five minutes against 14 calls in eleven minutes single-threaded.
#
# So: many seeds, few records each, a bounded number in flight.
#
# Usage:
#   bash scripts/gen_paper_fleet.sh <tier> <seeds> [per_seed] [parallel]
#   bash scripts/gen_paper_fleet.sh medium 200 3 24
set -uo pipefail

TIER="${1:?usage: gen_paper_fleet.sh <tier> <seeds> [per_seed] [parallel]}"
SEEDS="${2:?usage: gen_paper_fleet.sh <tier> <seeds> [per_seed] [parallel]}"
PER_SEED="${3:-3}"
PARALLEL="${4:-24}"
MODEL="${MODEL:-qwen/qwen3.7-flash}"
OUT_DIR="artifacts/paper_${TIER}"
CFG_DIR="configs/_fleet_${TIER}"

mkdir -p "$OUT_DIR" "$CFG_DIR"
echo "tier=$TIER seeds=$SEEDS per_seed=$PER_SEED parallel=$PARALLEL model=$MODEL"
echo "target: $((SEEDS * PER_SEED)) records into $OUT_DIR"

running=0
for i in $(seq 1 "$SEEDS"); do
  seed=$(( 100000 + i * 7919 ))          # coprime stride: distinct pools per seed
  cfg="$CFG_DIR/s$i.yaml"
  out="$OUT_DIR/s$i.jsonl"
  [ -s "$out" ] && continue              # resume: keep what a stopped run finished

  sed -e "s|^  seed: .*|  seed: $seed|" \
      -e "s|^  count: .*|  count: $PER_SEED|" \
      -e "s|^  max_llm_calls: .*|  max_llm_calls: $((PER_SEED * 12))|" \
      -e "s|^  model_id: .*|  model_id: $MODEL|" \
      configs/gen_paper_probe.yaml > "$cfg"
  # `tier` steers `target_count` in paper_pipeline: easy 1 table, medium 2,
  # intermediate 3, hard 3. It is the only knob that produces derived-tier data,
  # and derived tiers are 64.3% of the exam and 0% of our training set.
  sed -i "s|^  tier: .*|  tier: $TIER|" "$cfg"

  PYTHONIOENCODING=utf-8 python scripts/run_official_generate.py \
      --config "$cfg" --count "$PER_SEED" --workers 1 \
      --usage-log "$OUT_DIR/usage_s$i.jsonl" \
      --output "$out" > "$OUT_DIR/s$i.log" 2>&1 &

  running=$((running + 1))
  # Stagger the launches. Sixteen processes starting together got
  # "qwen/qwen3.7-flash is temporarily rate-limited upstream" (HTTP 429) and the
  # whole fleet died after eighteen seeds with two records. The provider limit is
  # on request rate, not on concurrency, so spacing the starts is enough.
  sleep 3
  if [ "$running" -ge "$PARALLEL" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
  if [ $((i % 20)) -eq 0 ]; then
    have=$(cat "$OUT_DIR"/s*.jsonl 2>/dev/null | grep -c . || echo 0)
    echo "  launched $i/$SEEDS seeds, $have records so far"
  fi
done
wait

have=$(cat "$OUT_DIR"/s*.jsonl 2>/dev/null | grep -c . || echo 0)
echo "done: $have records in $OUT_DIR"
echo "merge with: cat $OUT_DIR/s*.jsonl > artifacts/paper_${TIER}_all.jsonl"
