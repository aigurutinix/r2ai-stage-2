# Architecture and design

This repository is the reproducible ViFinQA submission pipeline. It keeps
retrieval, evidence packaging, and answer execution separate so each stage can
be checked independently.

## Data flow

```text
organizer questions + financial statements
        |
        v
document gate: ticker, year, scope, filing eligibility
        |
        v
candidate retrieval: BM25 + same-triple dense + graph expansion
        |
        v
question-level candidate union
        |
        v
Qwen3 reranker scores every exported fact/table pair
        |
        v
RRF fusion + adaptive gap-3 table ordering
        |
        v
source table materialization and evidence CSVs
        |
        v
grounded Qwen3.5 answer generation
        |
        v
allowlisted sandbox replay + submission validation
        |
        v
submission.zip
```

## Responsibilities

`run.py` is the package entrypoint. It parses questions, applies the document
gate, retrieves tables, binds candidate cells, writes source-derived evidence,
and creates the submission package.

`src/docs.py` owns organizer-format loading and deterministic question/report
metadata. `src/vifinqa/retrieval.py` parses tables and computes metadata-gated
BM25 retrieval. `src/vifinqa/fusion.py` combines sparse, dense, and reranker
orders. `src/vifinqa/rerank.py` provides the single table representation used
by pair export and GPU scoring. `src/vifinqa/answers.py` performs deterministic
row, year, unit, and numeric binding before the answer program is executed.

The `scripts/` directory contains stage boundaries and validators:

* candidate export, graph expansion, sidecar merge;
* reranker score validation and RRF/adaptive ranking;
* evidence/package validation and execution replay.

`kaggle/` contains only the two off-box GPU jobs: dense-index construction and
Qwen3 pair scoring. The notebooks are optional launch surfaces for reranking
and the Qwen3.5 answer stage; they do not define a second pipeline.

## Retrieval contract

The semantic fact query stays unchanged. Retrieval may widen the candidate
pool with dense matches and graph-linked tables, but aliases and sidecars do
not create new answer obligations. Every eligible candidate is scored before
adaptive selection; destructive pruning is post-score only.

The reranker changes ordering, not the document gate or accounting meaning.
Metadata filters remain deterministic. A missing or incomplete score manifest
fails validation instead of silently falling back to a partial ranking.

The adaptive gap-3 output is an ordering supplied to `run.py`; `run.py` still
materializes the exact tables and declares their source reports in the package.

## Answer contract

Answer generation receives only the evidence CSVs produced by the retrieval
package. The notebook asks Qwen3.5 for a grounded Pandas expression, validates
the expression against the allowlist, retries boundedly on invalid output, and
replays it in the sandbox. The numeric parser handles signs, decimal commas,
and source units before execution. Invalid evidence, unsafe code, missing
columns, or failed replay are errors—not reasons to invent a value.

## Reproducibility

Inputs and model weights stay outside Git. Each run uses a new output directory
and records manifests and hashes at stage boundaries. The committed ZIP is the
public-best control; a fresh retrieval run requires organizer data, the dense
index, and GPU-generated scores. Exact commands and notebook inputs are in
`README.md`.

## Design limits

This is intentionally a deterministic pipeline around two runtime model jobs:
table-pair reranking and grounded answer generation. Dense-index construction is
an explicit offline preparation step. The pipeline does not use planner agents,
recursive reflection, embeddings at answer time, or an LLM to choose a numeric
cell when table structure identifies that cell. Those would be separate
experiments, not hidden behavior in this submission.
