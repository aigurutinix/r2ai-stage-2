"""Build a submission with real answers for the questions we can decide.

Every emitted program is executed here, in the same sandbox the scorer uses, and
the reported `answer` is whatever that execution returned. A program that
crashes, or whose value disagrees with the lookup, is dropped rather than
shipped: `EXECUTION_ACCURACY` counts code that runs *and* reproduces the answer,
so a query we cannot re-run is worth no more than a placeholder and risks less.

Search depth and declared depth are separate knobs. Measured on the public
leaderboard: TABLES_F2 peaks at 5 declared refs (0.3817 vs 0.3457 at 10) because
precision collapses as k grows, while DOCS_F2 and ANSWER_ACCURACY both peak at
10 because a deeper search finds the figure more often. Searching deep and
declaring shallow takes the better half of each.

Usage:  python scripts/run_answer.py [search_k] [declare_k] [abs]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.corroborate import Corroborator  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.corpus.numeric import is_correct  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402
from vifin.submit.package import Prediction, TableRefStyle, build_submission  # noqa: E402
from vifin.submit.validate import validate_submission  # noqa: E402

PLACEHOLDER_QUERY = "result = 0.0"


def main() -> None:
    search_k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    declare_k = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 5
    magnitude = "abs" in sys.argv[1:]
    root = Path(__file__).resolve().parents[1]
    started = time.time()

    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    corroborator = Corroborator(store)

    predictions: list[Prediction] = []
    stats = {"answered": 0, "no_unit": 0, "no_match": 0, "crashed": 0, "mismatch": 0,
             "answered_lookup": 0, "answered_derived": 0, "single": 0, "confirmed": 0}

    for question in parsed:
        hits = retriever.search(question, top_k=search_k)
        searched = [hit.key for hit in hits]
        refs = searched[:declare_k]
        scale_out = question.unit_scale
        single = lookup_mod.is_single_lookup(question.question)
        stats["single"] += int(single)

        answer = 0.0
        query = PLACEHOLDER_QUERY
        evidence = refs[:1]

        if scale_out is None:
            stats["no_unit"] += 1
        else:
            picked = corroborator.choose(question, searched)
            if picked is None:
                stats["no_match"] += 1
            else:
                key = picked.key
                grid = store.rows(key)
                meta = store.meta(key)
                found = lookup_mod.find(grid, question)
                scale_in = lookup_mod.column_scale(
                    grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
                )
                code = lookup_mod.synthesize(found, scale_in, scale_out, magnitude=magnitude)
                outcome = run_query(code, {"df": grid})
                raw = abs(found.value) if magnitude else found.value
                expected = round(raw * scale_in / scale_out, 2)
                if not outcome.ok:
                    stats["crashed"] += 1
                elif not is_correct(expected, outcome.value):
                    stats["mismatch"] += 1
                else:
                    answer, query, evidence = outcome.value, code, [key]
                    stats["answered"] += 1
                    stats["answered_lookup" if single else "answered_derived"] += 1
                    stats["confirmed"] += int(picked.confirmed)

        predictions.append(
            Prediction(
                id=question.id,
                question=question.question,
                answer=answer,
                pandas_query=query,
                tables=evidence,
                ref_tables=refs,
                ref_docs=[k.doc_name for k in searched],
            )
        )

    total = len(parsed)
    print(f"search_k={search_k} declare_k={declare_k}  {total} questions in {time.time() - started:.1f}s")
    print(f"  answered deterministically  {stats['answered']} ({stats['answered'] / total:.1%})")
    print(f"  skipped, no currency unit   {stats['no_unit']}")
    print(f"  no label match in top-{search_k}     {stats['no_match']}")
    print(f"  program crashed             {stats['crashed']}")
    print(f"  value disagreed             {stats['mismatch']}")
    print(f"  single-lookup questions     {stats['single']}")
    print(f"    of which answered         {stats['answered_lookup']}"
          f"  ({stats['answered_lookup'] / max(1, stats['single']):.1%} coverage)")
    print(f"  derived questions answered  {stats['answered_derived']} (likely wrong, kept as free guesses)")
    print(f"  corroborated by next year   {stats['confirmed']}"
          f"  ({stats['confirmed'] / max(1, stats['answered']):.1%} of answers)")

    suffix = "_abs" if magnitude else "_raw"
    out = root / "submissions" / f"answer_s{search_k}d{declare_k}{suffix}.zip"
    summary = build_submission(predictions, store, out, style=TableRefStyle())
    problems = validate_submission(out, {p.id for p in parsed})
    print(f"\n{out.name}: {summary['csv_files']} csv, {summary['bytes'] / 1e6:.1f} MB, "
          f"{'OK' if not problems else str(problems[:3])}")


if __name__ == "__main__":
    main()
