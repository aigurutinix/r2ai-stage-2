"""P9 driver: build probe submissions.

Two unknowns are being measured, and the leaderboard happens to separate them:

1. *Which questions the public phase grades.* The scorer rejected a 1,012-row
   prediction with `gold=506 pred=1012`, so the public split is half the released
   set and no file names which half. `DOCS_F2MACRO` answers this — it depends
   only on `relevant_docs`, and our document filter is accurate enough that the
   right subset scores clearly above zero while a wrong one lands near it.

2. *How `relevant_tables` positions are written.* The organisers' code builds
   `f"{doc}|table_{id}"`; the published rules show `doc|350`. Nothing states
   whether ids are 0- or 1-based. `TABLES_F2MACRO` answers this, and it is
   readable independently of (1) because `DOCS_F2MACRO` ignores the convention.

Usage:
    python scripts/run_probe.py first506 table_0 bare0
    python scripts/run_probe.py odd506 table_0
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.query.parse import ParsedQuestion, parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402
from vifin.submit.package import Prediction, TableRefStyle, build_submission  # noqa: E402
from vifin.submit.validate import validate_submission  # noqa: E402

TOP_K = 5
PUBLIC_SPLIT_SIZE = 506
PLACEHOLDER_QUERY = "result = 0.0"

STYLES = {
    # Settled by the organisers: bare 1-based start line, e.g. `doc|350`.
    "line": TableRefStyle(prefixed=False, offset=0),
    # Kept only to re-measure the alternatives that scored zero.
    "line_minus1": TableRefStyle(prefixed=False, offset=-1),
    "prefixed": TableRefStyle(prefixed=True, offset=0),
}


def select(parsed: list[ParsedQuestion], subset: str) -> list[ParsedQuestion]:
    ordered = sorted(parsed, key=lambda p: p.id)
    if subset == "all":
        return ordered
    if subset == "first506":
        return ordered[:PUBLIC_SPLIT_SIZE]
    if subset == "last506":
        return ordered[-PUBLIC_SPLIT_SIZE:]
    if subset == "odd506":
        return [p for p in ordered if p.id % 2 == 1]
    if subset == "even506":
        return [p for p in ordered if p.id % 2 == 0]
    raise SystemExit(f"unknown subset: {subset}")


def main() -> None:
    args = sys.argv[1:] or ["first506", "table_0"]
    subset, style_names = args[0], args[1:] or ["table_0"]

    root = Path(__file__).resolve().parents[1]
    started = time.time()

    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    chosen = select(parsed, subset)
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    print(f"subset={subset}  {len(chosen)} questions  (ids {chosen[0].id}..{chosen[-1].id})")

    predictions = []
    for question in chosen:
        hits = retriever.search(question, top_k=TOP_K)
        predictions.append(
            Prediction(
                id=question.id,
                question=question.question,
                answer=0.0,
                pandas_query=PLACEHOLDER_QUERY,
                tables=[hit.key for hit in hits],
            )
        )
    print(f"retrieval took {time.time() - started:.1f}s")

    expected = {p.id for p in chosen}
    for name in style_names:
        style = STYLES[name]
        out = root / "submissions" / f"probe_{subset}_{name}.zip"
        summary = build_submission(predictions, store, out, style=style)
        problems = validate_submission(out, expected)
        status = "OK" if not problems else f"{len(problems)} PROBLEMS"
        print(f"  {out.name:34s} {summary['csv_files']:5d} csv  {summary['bytes'] / 1e6:5.1f} MB  {status}")
        for problem in problems[:5]:
            print(f"      - {problem}")


if __name__ == "__main__":
    main()
