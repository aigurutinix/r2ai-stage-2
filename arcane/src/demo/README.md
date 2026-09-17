# Demo day

A presenter-facing chat UI over the ViFinQA pipeline. A fresh Vietnamese
question runs the real stages — document gate, BM25 (plus optional dense),
reranking, gap-3 selection, grounded answer, sandboxed execution — and every
number shown is traced back to the source cell it came from.

## Run it

```bash
python -m pip install -r demo/requirements.txt
export DEMO_MODE=fixture
demo/launch_demo.sh
```

Demo dependencies live in `demo/requirements.txt` rather than in the project's
`pyproject.toml`, because nothing outside this directory needs them and the
submission methodology's own files are left untouched.

Against a rented GPU, one command does the whole thing; see
[REHEARSAL_VAST.md](REHEARSAL_VAST.md).

```bash
demo/vast_up.sh 'ssh -p PORT root@HOST'
```

For the live pipeline, set `VIFINQA_DATA_ROOT`, `PLANNER_URL`, and
`RERANK_URL` before the same launch command. The launcher uses the current
Python environment and never stops an existing process; use `DEMO_PORT` if
port 8000 is occupied.

`DEMO_MODE=fixture` runs the UI with a labelled placeholder answer and no
models. The banner says so on screen; never present from it.

## Pre-demo check without a GPU

```bash
VIFINQA_DATA_ROOT=../data/raw/vifinqa PYTHONPATH=src:. python demo/smoke_local.py
```

Stubs only the two model calls. Everything else is the real path, so a failure
here is a corpus or pipeline problem, not a serving problem.

## Development tier (no GPU)

`demo/dev_server.py` serves the same two interfaces from the 0.6B tier, so the
whole pipeline can be exercised on a laptop:

```bash
PYTHONPATH=src:. python -m uvicorn demo.dev_server:app --port 8001
PLANNER_URL=http://127.0.0.1:8001 RERANK_URL=http://127.0.0.1:8001 \
  PLANNER_MODEL=Qwen/Qwen2.5-0.5B-Instruct RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B \
  MODEL_TIMEOUT_SECONDS=3600 demo/launch_demo.sh
```

`MODEL_TIMEOUT_SECONDS` defaults to 120, which suits a GPU. On CPU the 0.6B
reranker needs several minutes for 50 candidates, so raise it or every question
times out mid-rerank.

The smoke path is designed and tested with model-call stubs so the gate,
retrieval, reranking, selection, execution, and citation stages can be checked
without a GPU. A live CPU measurement is not part of the current demo
verification; the demo runs Qwen3.5-9B for presenter use.

## Optional dense arm

Off unless both variables are set:

```bash
export EMBEDDING_URL=http://127.0.0.1:8003  # vLLM serving Qwen3-Embedding-4B
export DENSE_ROOT=../output/dense
```

The stored index is only valid for `Qwen/Qwen3-Embedding-4B`; the pipeline
refuses a mismatched served model rather than querying it with the wrong
vectors. Two caveats before you spend the VRAM:

- The index covers 1,829 of 1,973 reports. Questions about an uncovered filing
  get no dense candidates and fall back to sparse silently; `/api/health`
  reports the coverage and the first uncovered reports.
- On the graded submission the dense arm supplied 21 of 5,877 selected tables
  (0.36%). It is recall insurance the reranker mostly declines.

## API

- `POST /api/chat` — `{"question": "..."}`, streams NDJSON stage events, one at a time.
- `GET /api/evidence/{request_id}/{citation_id}` — the cited source table and the highlighted cell.
- `GET /api/health` — corpus counts, endpoint probes, local execution check, dense coverage.

## What the answer stage guarantees

The model picks cells; Python compiles and runs the expression. Citations are
derived from the `iloc` coordinates the executed query actually read, so a cell
that was not used cannot be cited, and a cited cell cannot be invented. A single
cell lookup is recompiled through `compile_lookup_query` so the table's own unit
reaches the unit the question asked for. One failed query gets exactly one
repair attempt with the error text, then the request fails visibly.

Source traceability and execution success are shown separately, and neither is a
claim about accounting interpretation.
