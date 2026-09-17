"""Re-render our corpus in the layout the organisers' generator expects.

`vifinqa catalog` reads our data root and reports 1,973 documents and **zero
tables**, because the two corpora encode tables differently. Their loader wants

    <root>/<ticker>/<year>/<doc_name>/
        <doc_name>_extracted.txt          # tables appear as [table_N](path) anchors
        <doc_name>_extracted_tables/
            table_0.csv, table_1.csv, ...

while the public corpus embeds `<table>...</table>` HTML inline. That is the
"public corpus is not their internal corpus" trap from IMPLEMENTATION.md, and it
is the only thing standing between us and running their question generator — the
generator that emits `QARecord`, which carries `pandas_query` and `answer` and is
therefore complete supervision for fine-tuning.

Nothing here re-parses anything. Every table was parsed once into
`artifacts/tables.parquet` with its document, its ordinal and its grid; this
writes those grids out as CSV and swaps each HTML block for the anchor that
points at them, preserving the `===== PAGE N =====` structure their document
parser keys on.

Writes to a NEW root. The existing corpus is the input to every other artifact we
have and is not modified.

Usage:
  PYTHONPATH=src python scripts/build_official_corpus.py [--limit N] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

TABLE_BLOCK_RE = re.compile(r"<table.*?</table>", re.S)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/official_corpus")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many documents (0 = all)")
    args = parser.parse_args()

    out_root = ROOT / args.out
    frame = pd.read_parquet(
        ROOT / "artifacts" / "tables.parquet",
        columns=["doc_name", "ticker", "year", "table_id", "rows_json"])
    # `table_id` is the ordinal of the table within its document, assigned in
    # reading order — the same order `TABLE_BLOCK_RE` finds the HTML blocks in.
    grids: dict[str, dict[int, list[list[str]]]] = {}
    for row in frame.itertuples(index=False):
        grids.setdefault(str(row.doc_name), {})[int(row.table_id)] = json.loads(row.rows_json)
    print(f"{len(grids)} documents with parsed tables in tables.parquet")

    source_root = ROOT / "data" / "financial_statements"
    started = time.time()
    docs = written_tables = missing_grid = mismatched = 0

    for ticker_dir in sorted(source_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for year_dir in sorted(ticker_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for doc_dir in sorted(year_dir.iterdir()):
                if not doc_dir.is_dir():
                    continue
                text_path = next(iter(doc_dir.glob("*_extracted.txt")), None)
                if text_path is None:
                    continue
                doc_name = doc_dir.name
                by_id = grids.get(doc_name)
                if not by_id:
                    missing_grid += 1
                    continue

                target_dir = out_root / ticker_dir.name / year_dir.name / doc_name
                tables_dir = target_dir / f"{doc_name}_extracted_tables"
                tables_dir.mkdir(parents=True, exist_ok=True)

                text = text_path.read_text(encoding="utf-8")
                ordinal = 0
                emitted: list[int] = []

                def swap(match: re.Match[str]) -> str:
                    nonlocal ordinal
                    table_id = ordinal
                    ordinal += 1
                    emitted.append(table_id)
                    return (f"[table_{table_id}]"
                            f"({doc_name}_extracted_tables/table_{table_id}.csv)")

                rewritten = TABLE_BLOCK_RE.sub(swap, text)
                if ordinal != len(by_id):
                    # The ordinals must line up with the parquet, or a question
                    # generated against `table_7` would cite a different table
                    # than the one our own pipeline calls `table_7`.
                    mismatched += 1

                for table_id in emitted:
                    grid = by_id.get(table_id)
                    if grid is None:
                        continue
                    csv_path = tables_dir / f"table_{table_id}.csv"
                    with csv_path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.writer(handle, lineterminator="\n")
                        writer.writerows(grid)
                    written_tables += 1

                (target_dir / f"{doc_name}_extracted.txt").write_text(
                    rewritten, encoding="utf-8")
                docs += 1
                if args.limit and docs >= args.limit:
                    break
            if args.limit and docs >= args.limit:
                break
        if args.limit and docs >= args.limit:
            break

    print(f"\nwrote {docs} documents, {written_tables} table CSVs "
          f"in {time.time() - started:.0f}s")
    print(f"  documents with no parsed grid  : {missing_grid}")
    print(f"  HTML block count != parquet rows: {mismatched}")
    print(f"  root: {out_root}")
    print("\nverify with:")
    print("  PYTHONPATH=vifinqa-official/src python -c \"import sys;"
          "sys.argv=['vifinqa','catalog','--data-root','"
          f"{args.out}'];from vifinqa.cli import main;main()\"")


if __name__ == "__main__":
    main()
