"""Which branch produced each constant-assignment program?

The build from the code we control needs `fix_constants` on 184 of 1012 programs;
the frozen artifact needs it on 4. A constant is a program that reads no cell, so
those 184 answers are not derived from the corpus at all — they are 18% of the exam
answered by fabrication, and the private round rejects them on manual review.

184 is far too many to be the 84 questions no branch claims. So some branch is
emitting a value without a cell read, and knowing which one turns a vague
"the rebuild is degraded" into a named defect in code we can change.

Re-runs the pipeline's branch selection and records `source` for every question
whose program reads nothing.

Usage:  PYTHONPATH=src python scripts/_constant_source.py --base _repro.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONSTANT = re.compile(r"^\s*result\s*=\s*-?[\d_.,eE+\-]+\s*$", re.MULTILINE)
READS_CELL = re.compile(r"num\s*\(|\.iloc\[|\.loc\[|df\d*\[")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="_repro.zip")
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))

    dead = []
    for row in rows:
        query = row.get("pandas_query") or ""
        if not query.strip() or not READS_CELL.search(query):
            dead.append(row)

    print(f"{len(dead)}/{len(rows)} chuong trinh khong doc o nao")

    shapes: Counter[str] = Counter()
    for row in dead:
        query = (row.get("pandas_query") or "").strip()
        if not query:
            shapes["rong"] += 1
        elif CONSTANT.match(query):
            shapes["result = <so>"] += 1
        else:
            shapes["khac"] += 1
    for name, count in shapes.most_common():
        print(f"  {name}: {count}")

    # Evidence tells which branch ran: a branch that located a cell declares the
    # table it read, and one that produced a number from nothing declares whatever
    # the retrieval offered.
    no_evidence = sum(1 for row in dead if not (row.get("evidence") or []))
    zero = sum(1 for row in dead if row.get("answer") in (0, 0.0))
    print(f"  trong so do: khong co evidence {no_evidence}, dap an bang 0 {zero}")

    print(f"\n{min(args.show, len(dead))} vi du:")
    for row in dead[:args.show]:
        query = (row.get("pandas_query") or "").strip().replace("\n", " ")[:70]
        print(f"  id={row['id']:4d} answer={row.get('answer')}  query={query!r}")
        print(f"     {row['question'][:100]}")


if __name__ == "__main__":
    main()
