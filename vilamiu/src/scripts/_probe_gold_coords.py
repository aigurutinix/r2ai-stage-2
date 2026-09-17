"""Compare the cell our deterministic path picks with the cell gold actually read.

The gold records carry the organisers' own `pandas_query`, and those queries name
the cell outright — `result = df.iloc[-1]['31/12/2024Triệu VND']`. So the row and the
column gold used are recoverable, and our own choice can be compared against each
separately. No model, no cost.

This has never been measured for the deterministic path. Every previous split of row
error against column error came from the LLM probe, and that probe was handed the
gold table, which is not the path a submission takes.

The distinction decides where work goes. A wrong row means the label matcher picked
the wrong line item; a wrong column means it found the item and read the wrong period
or the wrong scope. Those need opposite fixes, and one of them may be most of the
残り.

Usage:
  PYTHONPATH=src python scripts/_probe_gold_coords.py --limit 900
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# The three shapes the generator emits.
ILOC_RC = re.compile(r"\.iloc\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
ILOC_NAME = re.compile(r"\.iloc\[\s*(-?\d+)\s*\]\s*\[\s*(['\"])(.*?)\2\s*\]", re.S)
LOC_NAME = re.compile(r"\.loc\[\s*(-?\d+)\s*,\s*(['\"])(.*?)\2\s*\]", re.S)


def gold_cell(query: str, grid):
    """(row, column) in grid coordinates, or None when the shape is unrecognised.

    A DataFrame built by `frame_from_rows` consumes grid row 0 as the header, so
    DataFrame row j is grid row j+1. A negative index counts from the end of the
    body, which in grid terms is len(grid) + index.
    """

    def to_grid_row(index: int) -> int:
        return len(grid) + index if index < 0 else index + 1

    match = ILOC_RC.search(query)
    if match:
        row = to_grid_row(int(match.group(1)))
        column = int(match.group(2))
        if column < 0:
            column = len(grid[0]) + column
        return row, column

    for pattern in (ILOC_NAME, LOC_NAME):
        match = pattern.search(query)
        if not match:
            continue
        row = to_grid_row(int(match.group(1)))
        name = match.group(3)
        header = [str(cell) for cell in grid[0]]
        if name in header:
            return row, header.index(name)
        # The extractor sometimes differs from the query by whitespace alone.
        squashed = [re.sub(r"\s+", "", cell) for cell in header]
        target = re.sub(r"\s+", "", name)
        if target in squashed:
            return row, squashed.index(target)
        return row, None
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=900)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    tally: collections.Counter[str] = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    seen = 0

    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or seen >= args.limit:
            continue
        record = json.loads(line)
        query = record.get("pandas_query") or ""
        refs = record.get("relevant_tables") or []
        if not query or len(refs) != 1:
            continue
        doc, _, table_id = refs[0].rpartition("|table_")
        if not table_id.isdigit():
            continue
        grid = store.rows(TableKey(doc, int(table_id)))
        if not grid or len(grid) < 2:
            continue
        located = gold_cell(query, grid)
        if located is None:
            tally["không đọc được query gold"] += 1
            continue
        gold_row, gold_col = located
        seen += 1

        # Our own choice, on the gold table, so only the reading step is measured.
        metric = lookup_mod.extract_metric(record["question"])
        found = lookup_mod.match_row(grid, metric)
        if found is None:
            tally["ta KHÔNG có ứng viên"] += 1
            continue
        our_row = found[0]

        if our_row == gold_row:
            tally["ĐÚNG dòng"] += 1
        elif abs(our_row - gold_row) == 1:
            tally["lệch 1 dòng"] += 1
            if len(examples["lệch 1 dòng"]) < args.show:
                examples["lệch 1 dòng"].append(
                    (record.get("id"), metric[:40],
                     str(grid[gold_row][0])[:40] if gold_row < len(grid) else "?",
                     str(grid[our_row][0])[:40]))
        else:
            tally["SAI dòng"] += 1
            if len(examples["SAI dòng"]) < args.show:
                examples["SAI dòng"].append(
                    (record.get("id"), metric[:40],
                     str(grid[gold_row][0])[:40] if gold_row < len(grid) else "?",
                     str(grid[our_row][0])[:40]))

    print(f"{seen} bản ghi gold đọc được toạ độ\n")
    total = max(sum(v for k, v in tally.items()
                    if k != "không đọc được query gold"), 1)
    for name, count in tally.most_common():
        print(f"  {name:26s} {count:5d}  {count / total:6.1%}")
    print()
    for name in ("lệch 1 dòng", "SAI dòng"):
        for qid, metric, gold_label, our_label in examples.get(name, []):
            print(f"  [{name}] id={qid} tra {metric!r}")
            print(f"        gold: {gold_label!r}")
            print(f"        ta  : {our_label!r}")


if __name__ == "__main__":
    main()
