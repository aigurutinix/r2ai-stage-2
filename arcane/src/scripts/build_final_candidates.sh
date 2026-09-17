#!/usr/bin/env bash
# Rebuild the complete sparse+dense+graph pool from organizer data and a dense index.
set -euo pipefail

cd "$(dirname "$0")/.."

FINAL_DIR=${1:?usage: scripts/build_final_candidates.sh OUTPUT_DIR}
PYTHON=${PYTHON:-.venv/bin/python}
DENSE_DIR=${DENSE_DIR:-output/dense}
QUESTIONS=data/raw/vifinqa/questions/questions.jsonl

if [[ -e "$FINAL_DIR" ]] && [[ -n "$(find "$FINAL_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty output directory: $FINAL_DIR" >&2
  exit 1
fi
mkdir -p "$FINAL_DIR"

"$PYTHON" -m pytest tests -q
"$PYTHON" scripts/build_line_item_lexicon.py --progress-every 200
"$PYTHON" scripts/verify_dense_artifacts.py \
  --dense "$DENSE_DIR" \
  --tables "$DENSE_DIR/tables.jsonl" \
  --questions "$QUESTIONS" \
  --manifest "$FINAL_DIR/dense_manifest.json"

"$PYTHON" scripts/export_rerank_pairs.py \
  --depth 50 \
  --dense "$DENSE_DIR" \
  --dense-depth 40 \
  --dense-policy same-triple \
  --hierarchy \
  --output "$FINAL_DIR/pairs_hybrid.jsonl" \
  --dense-tables "$FINAL_DIR/dense_tables.json" \
  --progress-every 100

"$PYTHON" scripts/apply_rerank_scores.py \
  --pairs "$FINAL_DIR/pairs_hybrid.jsonl" \
  --first-stage-only \
  --arm hybrid \
  --output "$FINAL_DIR/ranking_hybrid_first_stage.json"

"$PYTHON" scripts/add_graph_tables.py \
  --pairs "$FINAL_DIR/pairs_hybrid.jsonl" \
  --base-ranking "$FINAL_DIR/ranking_hybrid_first_stage.json" \
  --arm hybrid \
  --depth 30 \
  --max-added 3 \
  --hierarchy \
  --output "$FINAL_DIR/ranking_graph_candidates.json" \
  --trace "$FINAL_DIR/graph.trace.jsonl" \
  --expanded-pairs "$FINAL_DIR/pairs_hybrid_graph.jsonl"

"$PYTHON" scripts/merge_table_sidecars.py \
  --input "$FINAL_DIR/dense_tables.json" \
  --input "$FINAL_DIR/ranking_graph_candidates.graph_tables.json" \
  --output "$FINAL_DIR/extra_tables.json"

"$PYTHON" scripts/validate_rerank_pairs.py \
  --pairs "$FINAL_DIR/pairs_hybrid_graph.jsonl" \
  --extra-tables "$FINAL_DIR/extra_tables.json" \
  --graph-trace "$FINAL_DIR/graph.trace.jsonl" \
  --questions "$QUESTIONS" \
  --dense-manifest "$FINAL_DIR/dense_manifest.json" \
  --manifest "$FINAL_DIR/candidate_manifest.json"

sha256sum \
  "$FINAL_DIR/pairs_hybrid_graph.jsonl" \
  "$FINAL_DIR/extra_tables.json" \
  "$FINAL_DIR/candidate_manifest.json"
