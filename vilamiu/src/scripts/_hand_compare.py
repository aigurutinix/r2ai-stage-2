"""Side-by-side dump of the questions where the reader and the shipped build differ.

Neither number can be checked against gold â€” the local gold set lies about
retrieval and has a unit defect in 28% of its records â€” so the only honest way to
call these is to look at the actual cell each side read: which table, which row
label, which column header, what the raw text was. This prints exactly that, so a
sample of the 285 disagreements can be judged by hand before a submission slot is
spent on them.

Usage:
  PYTHONPATH=src python scripts/_hand_compare.py --n 12 --seed 7
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.store import TableKey, TableStore  # noqa: E402

CELL = re.compile(r"num\(\s*(\w+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def load_zip(name: str) -> dict[int, dict]:
    with zipfile.ZipFile(ROOT / "submissions" / name) as archive:
        return {r["id"]: r for r in json.loads(archive.read("submission.json"))}


def cells(query: str) -> list[tuple[str, int, int]]:
    return [(m.group(1), int(m.group(2)), int(m.group(3))) for m in CELL.finditer(query)]


def describe(row: dict, store: TableStore, keys: list[TableKey] | None = None) -> list[str]:
    """One line per cell the program reads: table, row label, column header, text."""

    by_var: dict[str, TableKey] = {}
    for item in row.get("evidence") or []:
        name = Path(item["csv_path"]).stem
        doc, _, tid = name.rpartition("_table_")
        try:
            by_var[item["variable"]] = TableKey(doc, int(tid))
        except ValueError:
            continue

    out = []
    for variable, r, c in cells(row.get("pandas_query") or "")[:6]:
        key = by_var.get(variable)
        if key is None:
            out.append(f"    {variable}[{r},{c}] (khong ro bang)")
            continue
        grid = store.rows(key)
        # `frame_from_rows` consumes grid row 0 as the header, so DataFrame row r
        # is grid row r+1. Reading grid[r] here printed the label one row above the
        # cell the program actually reads, which made every judgement wrong.
        g = r + 1 if r >= 0 else r
        if not (-len(grid) <= g < len(grid)):
            out.append(f"    {key.doc_name} t{key.table_id} [df{r},{c}] NGOAI BANG")
            continue
        line = grid[g]
        label = str(line[0])[:60] if line else ""
        header = str(grid[0][c])[:40] if grid and c < len(grid[0]) else ""
        text = str(line[c])[:28] if c < len(line) else "NGOAI COT"
        out.append(f"    t{key.table_id} df{r} '{label}' | cot{c} '{header}' = {text}")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="aimed.zip")
    parser.add_argument("--other", default="ts_cell.zip")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base = load_zip(args.base)
    other = load_zip(args.other)
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    differ = [i for i in sorted(base)
              if str(base[i].get("answer")) != str(other[i].get("answer"))]
    random.Random(args.seed).shuffle(differ)
    print(f"{len(differ)} cau khac nhau; in {min(args.n, len(differ))} cau\n")
    for qid in differ[:args.n]:
        print(f"### id={qid}  {base[qid]['question']}")
        print(f"  aimed = {base[qid].get('answer')}")
        for line in describe(base[qid], store):
            print(line)
        print(f"  may doc = {other[qid].get('answer')}")
        for line in describe(other[qid], store):
            print(line)
        print()


if __name__ == "__main__":
    main()

