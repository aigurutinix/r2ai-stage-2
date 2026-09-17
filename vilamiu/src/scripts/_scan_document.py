"""Stop ranking tables. The document is known — scan every table inside it.

The evidence from 20/08 converges on one explanation. Nine independent channels fail
on the same 421 questions. The tables those questions retrieve are structurally
indistinguishable from the tables the easy questions retrieve — same row count, same
column count, same empty-cell density, no raggedness, same header depth. Deeper
search made things worse, model cell-picking scored near zero, and ranking by label
score is anti-correlated with correctness.

Mechanisms that differ this much do not fail identically unless they share an input.
They do: all of them read from the same shortlist of 8 to 30 tables, produced by a
lexical ranking over whole-table text. A statement table is mostly rows that have
nothing to do with the question, so whole-table similarity is dominated by noise —
and the one row that matters contributes a few tokens out of hundreds.

But `DOCS_RECALL` is 0.955. The document is not the problem; the ranking inside it
is. And a document holds about 74 tables (146,246 across 1,965 documents), which is
nothing to scan exhaustively. So the ranking can be deleted rather than improved:
take the ticker, the year and the scope the question names, and match its metric
against every row label of every table in those documents.

This is not "search deeper". Deeper in a bad ranking adds bad tables. This removes
the ranking from the path entirely for the one step where it was doing harm.

Reports the reads the shortlist misses, and whether the tables it finds are ones the
shortlist never offered.

Usage:  PYTHONPATH=src python scripts/_scan_document.py --n 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SEPARATE = ("công ty mẹ", "riêng", "cong ty me")
CONSOLIDATED = ("hợp nhất", "hop nhat")


def wanted_scope(question: str) -> str | None:
    text = question.casefold()
    if any(word in text for word in SEPARATE):
        return "separate"
    if any(word in text for word in CONSOLIDATED):
        return "consolidated"
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--consensus", default="artifacts/consensus_strong.jsonl")
    parser.add_argument("--out", default="artifacts/scan_doc.jsonl")
    parser.add_argument("--only-hard", action="store_true",
                        help="only questions where a single channel reads anything")
    args = parser.parse_args()

    clusters = {}
    path = ROOT / args.consensus
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                clusters[record["id"]] = record["size"]

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    frame = store.frame

    # One pass to group every table by (ticker, year, scope) so the scan is a
    # dictionary lookup rather than a filter over 146k rows per question.
    by_report: dict[tuple[str, str, str], list[TableKey]] = {}
    for ticker, year, scope, doc, table_id in zip(
            frame["ticker"], frame["year"], frame["scope"],
            frame["doc_name"], frame["table_id"]):
        by_report.setdefault((str(ticker), str(year), str(scope)), []).append(
            TableKey(str(doc), int(table_id)))
    sizes = sorted(len(v) for v in by_report.values())
    print(f"{len(by_report)} bao cao, so bang moi bao cao: "
          f"p50={sizes[len(sizes) // 2]} p90={sizes[9 * len(sizes) // 10]} "
          f"max={sizes[-1]}", flush=True)

    scope_questions = [q for q in questions if len(q.tickers) == 1 and q.years]
    if args.only_hard:
        scope_questions = [q for q in scope_questions
                           if clusters.get(q.id, 0) <= 1]
    scope_questions = scope_questions[:args.n]
    print(f"{len(scope_questions)} cau", flush=True)

    counters: Counter[str] = Counter()
    records = []
    started = time.time()
    for index, question in enumerate(scope_questions, start=1):
        metric = lookup_mod.extract_metric(question.question)
        year = str(max(question.years))
        scope = wanted_scope(question.question)
        keys: list[TableKey] = []
        for candidate_scope in (["separate", "consolidated"] if scope is None
                                else [scope]):
            keys += by_report.get((question.tickers[0], year, candidate_scope), [])
        if not keys:
            counters["khong co bao cao"] += 1
            continue

        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        shortlist = {hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)}

        best = None
        for key in keys:
            grid = store.rows(key)
            if len(grid) < 2:
                continue
            label_col = lookup_mod.label_column(grid)
            match = lookup_mod.match_row(grid, metric, label_col,
                                         store.meta(key).caption)
            if match is None:
                continue
            row, score, label = match
            if score < lookup_mod.MIN_LABEL_SCORE:
                continue
            if best is None or score > best[0]:
                best = (score, key, row, label)
        if best is None:
            counters["quet het van khong khop"] += 1
            continue
        counters["quet ra dong"] += 1
        score, key, row, label = best
        outside = key not in shortlist
        counters["bang NGOAI shortlist" if outside else "bang trong shortlist"] += 1

        # Emit a runnable program, not just a coordinate: the scorer re-runs
        # `pandas_query`, and a reading that cannot be expressed as one is not an
        # answer. Same emitter the shipped lookup branch uses, so the only
        # difference from it is which tables were considered.
        grid = store.rows(key)
        label_col = lookup_mod.label_column(grid)
        column = lookup_mod.pick_column(grid, question, label_col)
        cell = (lookup_mod._parse_cell(grid[row][column])
                if column is not None and column < len(grid[row]) else None)
        if column is None or cell is None:
            counters["khong doc duoc o"] += 1
            continue
        meta = store.meta(key)
        scale_in = lookup_mod.column_scale(
            grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}")
        code = lookup_mod.synthesize(
            lookup_mod.Lookup(row=row, column=column, label=label, value=cell,
                              score=score, label_col=label_col),
            scale_in, question.unit_scale or 1.0, magnitude=True)
        outcome = run_query(code, {"df": grid})
        if not outcome.ok:
            counters["chuong trinh loi"] += 1
            continue
        records.append({
            "id": question.id,
            "ok": True,
            "value": outcome.value,
            "error": "",
            "attempts": 1,
            "code": code,
            "variables": ["df"],
            "refs": {"df": f"{key.doc_name}|{meta.start_line}"},
            "keys": [[key.doc_name, key.table_id]],
            "labels": [label],
            "score": round(score, 3),
            "outside_shortlist": bool(outside),
            "n_tables_scanned": len(keys),
        })
        if index % 50 == 0:
            print(f"  {index}/{len(scope_questions)}  "
                  f"{time.time() - started:.0f}s", flush=True)

    (ROOT / args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    print()
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    if records:
        outside = sum(1 for r in records if r["outside_shortlist"])
        print(f"\ndong tot nhat nam trong bang MA SHORTLIST KHONG HE DE NGHI: "
              f"{outside}/{len(records)} ({100 * outside / len(records):.0f}%)")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
