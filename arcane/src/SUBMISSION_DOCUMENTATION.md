# ViFinQA Stage 2 — Product Documentation

## 1. Product summary

This submission answers Vietnamese financial questions by retrieving source
tables from the organizer corpus, selecting evidence tables, and executing a
grounded numerical program. Every submitted evidence CSV is materialized from
an organizer-provided table and retains document/table provenance.

## 2. Architecture

```text
organizer questions + reports
        ↓
company/year/scope eligibility
        ↓
parent + fact retrieval (BM25 and dense)
        ↓
Qwen3-Reranker-8B fact×table scoring
        ↓
RRF/adaptive-gap table selection
        ↓
source-derived evidence CSVs
        ↓
Qwen3.5-9B grounded answer program
        ↓
Pydantic validation → sandbox replay → submission.zip
```

The planner identifies raw evidence requirements. Retrieval decides which
eligible tables contain those requirements. The answer model may reference only
rows and cells shown in the evidence context; Python validates and executes the
result. No answer or financial value is hard-coded.

## 3. Data

The only financial data are the ViFinQA questions, company mapping, and
financial-statement reports supplied by the organizers. Expected local layout:

```text
data/raw/vifinqa/
├── code_stock.csv
├── questions/questions.jsonl
└── financial_statements/<ticker>/<year>/...
```

The organizer data must be shared to the review team through the approved
Google Drive, OneDrive, or equivalent access link. Insert the final access link
in the delivery email; it is intentionally not committed to this public repo.

## 4. Models and checkpoints

| Model | Revision | Use |
| --- | --- | --- |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` | dense table index |
| `Qwen/Qwen3-Reranker-8B` | `77d193c791ed757ca307ee72715aa132723da912` | fact/table reranking |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | grounded answer programs |

All models are public Hugging Face checkpoints. Download the pinned revisions
with `huggingface-cli download`; provide the corresponding Hugging Face links
in the delivery email.

## 5. Reproduction

```bash
git clone https://github.com/lducc/vifinqa-final-submission.git
cd vifinqa-final-submission
python -m pip install -e .
```

Place organizer data and the generated dense index in the paths described above.
Build candidates:

```bash
RUN=output/final_run
PYTHON=python DENSE_DIR=output/dense scripts/build_final_candidates.sh "$RUN"
```

Score `$RUN/pairs_hybrid_graph.jsonl` on a GPU with
`kaggle/rerank_qwen_8b.py`, then validate scores, apply the adaptive-gap
selection, and package the retrieval result:

```bash
python scripts/validate_rerank_scores.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores.jsonl" \
  --manifest "$RUN/scores.jsonl.manifest.json"
python scripts/apply_rerank_scores.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores.jsonl" \
  --arm hybrid --weight 0.5 \
  --output "$RUN/ranking_hybrid_qwen.json"
python scripts/select_slot_tables.py \
  --pairs "$RUN/pairs_hybrid_graph.jsonl" \
  --scores "$RUN/scores.jsonl" \
  --base-ranking "$RUN/ranking_hybrid_qwen.json" \
  --starting-ranking "$RUN/ranking_hybrid_qwen.json" \
  --adaptive-append --logit-gap 3 \
  --output "$RUN/ranking_adaptive_gap3.json"
```

Then run `run.py` with the selected ranking and validate the resulting package.
Create answer inputs from the retrieval ZIP:

```bash
python scripts/build_answer_inputs.py retrieval_fresh.zip \
  --output-dir /content/answer_inputs
```

Run every cell of `notebooks/final_answer.ipynb` on an A100. The notebook uses
checkpointed, deterministic Qwen3.5-9B generation, Pydantic program checks,
bounded repair, sandbox replay, and writes the final answer ZIP only after all
1,012 records validate.

## 6. Submission compliance

The final ZIP must contain evidence CSVs that are subsets of organizer data and
are traceable through `relevant_docs`, `relevant_tables`, and `evidence`.
Queries must compute results directly from those CSVs at execution time. Values
must not be assigned as constants, copied from stored answers, or encoded in
the query. Run:

```bash
python scripts/validate_submission.py path/to/package
python scripts/verify_execution.py --package path/to/package
```

Include this document, the source repository, dependency/configuration details,
the organizer-data access link, checkpoint links, and the SHA-256 of the final
`submission.zip` in the delivery dossier.

## 7. Known limitations

The answer stage is intentionally conservative: malformed programs are retried
with validation feedback and then rejected rather than replaced by a stored
answer. A run is complete only when all questions produce validated,
source-grounded execution records.
