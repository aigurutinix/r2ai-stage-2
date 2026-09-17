"""Are the hard questions hard because of the tables, not the mechanisms?

Everything measured on 20/08 points the same way. Nine independent channels, and on
421 of 1012 questions only ONE of them can read anything at all. Model cell-picking
scored ~0% on the disputed rows; deeper cross-table search scored worse than the
shallow one; ranking by label score is anti-correlated with being right. Mechanisms
as different as a regex matcher, a 14B program generator and a two-stage reader all
fail on the same questions.

When every mechanism fails together, the mechanism is not the variable. The
substrate is: `store.rows()` hands out a flat grid, and if the OCR produced a
merged header, a ragged body, or half a table that continues on the next page, no
mechanism can read a row out of it.

This compares the tables behind the questions consensus finds easy against the
tables behind the questions where only one channel reads anything, using only
structural features of the grid — nothing about the answer, no gold:

  n_rows, n_cols        a two-row stub or a 200-row run-on is not a clean table
  ragged                rows of differing length mean cells were lost or merged
  empty density         OCR dropping cells shows up here
  numeric density       a table of mostly text is a narrative, not a statement
  header depth          how many leading rows carry no figures
  continuation          a table whose first row already looks like data has lost
                        its header to the page break

If the two populations differ, table repair lifts every mechanism at once and is
worth more than any new answering branch.

Usage:  PYTHONPATH=src python scripts/_substrate.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def features(grid: list[list[str]]) -> dict[str, float]:
    rows = len(grid)
    if rows == 0:
        return {}
    widths = [len(r) for r in grid]
    cols = max(widths)
    cells = sum(widths) or 1
    empty = sum(1 for r in grid for c in r if not str(c).strip())
    numeric = sum(1 for r in grid for c in r
                  if lookup_mod._parse_cell(c) is not None)
    header_depth = 0
    for line in grid:
        if any(lookup_mod._parse_cell(c) is not None for c in line):
            break
        header_depth += 1
    label_col = lookup_mod.label_column(grid)
    value_cols = len(lookup_mod.value_columns(grid, label_col))
    return {
        "n_rows": rows,
        "n_cols": cols,
        "ragged": 1.0 if len(set(widths)) > 1 else 0.0,
        "empty": empty / cells,
        "numeric": numeric / cells,
        "header_depth": header_depth,
        "value_cols": value_cols,
        "no_value_col": 1.0 if value_cols == 0 else 0.0,
        "continuation": 1.0 if header_depth == 0 else 0.0,
    }


def summarise(name: str, samples: list[dict[str, float]]) -> None:
    if not samples:
        print(f"{name}: khong co mau")
        return
    keys = ("n_rows", "n_cols", "ragged", "empty", "numeric", "header_depth",
            "value_cols", "no_value_col", "continuation")
    print(f"\n{name}  (n={len(samples)})")
    for key in keys:
        values = [s[key] for s in samples if key in s]
        if not values:
            continue
        print(f"  {key:14s} trung vi {statistics.median(values):8.3f}   "
              f"trung binh {statistics.fmean(values):8.3f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus", default="artifacts/consensus_strong.jsonl")
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--easy", type=int, default=5, help="cluster size >= this")
    args = parser.parse_args()

    clusters = {}
    for line in (ROOT / args.consensus).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            clusters[record["id"]] = record["size"]

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    easy, hard = [], []
    for question in questions:
        size = clusters.get(question.id)
        if size is None:
            continue
        if size >= args.easy:
            bucket = easy
        elif size == 1:
            bucket = hard
        else:
            continue
        if len(question.tickers) != 1 or not question.years:
            continue
        groups = max(1, len(question.tickers)) * max(1, len(question.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            question, per_group=per_group, cap=args.shortlist)]
        for key in keys[:3]:
            data = features(store.rows(key))
            if data:
                bucket.append(data)

    summarise(f"DE  (>={args.easy} kenh dong thuan)", easy)
    summarise("KHO (chi 1 kenh doc duoc)", hard)


if __name__ == "__main__":
    main()
