# Package `road2ai_vifinqa`

## Entry points

- `build_index.py`: OCR/HTML → SQLite corpus và manifest.
- `build_panel.py`: SQLite → canonical financial panel.
- `solve.py`: route 1.012 câu, checkpoint, build ZIP.
- `validate.py`: validation CLI.

## Data và semantics

- `paths.py`: đường dẫn chuẩn của project/dataset/artifact/run.
- `corpus.py`: read-only API cho document/table/row assets.
- `html_tables.py`: parse và bung table spans.
- `table_semantics.py`: header, period, section, table type, cell semantics.
- `source_units.py`: nhận dạng/kế thừa unit có guard.
- `text.py`: normalize tiếng Việt và parse số OCR.
- `build_panel.py`, `panel.py`, `comparative_panel.py`: canonical metrics và
  comparative prior-year cells.

## Retrieval và solver

- `retrieval.py`, `direct.py`: lexical retrieval và direct-value baseline.
- `easy_reranker.py` + `easy_reranker_v2.json`: 58-feature linear shortlist ranker.
- `easy_solver.py`: exhaustive candidates + Qwen3-8B grounded selection.
- `hard_solver.py`: formula registry trên financial panel.
- `hard_note_solver.py`: disclosure/note retrieval và checked compiler.
- `template_solver.py`: deterministic generator-template recipes.
- `raw_solver.py`, `panel_solver.py`: generic fallback/research solvers.
- `expression_plan.py`: compile bounded expression từ grounded operands.

## Model/runtime

- `local_llm.py`: process manager + OpenAI-compatible llama.cpp client, no-think.
- `reasoning_llm.py`: opt-in thinking client cho source-only audits.

## Submission

- `pipeline.py`: adapter từ specialized solution sang `SubmissionSolution`.
- `submission.py`: evidence frames, query evaluation và ZIP writer.

Nguyên tắc dependency: parser/semantics không phụ thuộc solver; solver chỉ đọc
corpus/panel; adapter và release layer không được âm thầm sửa grounding. Chi tiết
end-to-end xem [`docs/architecture.md`](../../docs/architecture.md).
