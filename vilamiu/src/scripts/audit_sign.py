"""Count answers reported positive where the cell they came from is negative.

Vietnamese statements print costs, losses and reductions in parentheses, and this
project wraps operands in `abs()` for that reason: "chi phí tài chính" parses
negative in one report and positive in another, so summing three of them mixed
signs.

But the organisers' own parser reads `(1.234)` as **-1234**
(`common/numeric/parsing.py:40-51`), and their generator rounds the value without
taking a magnitude (`hard/recipe/compiler.py:54-56`). So where the cell is
parenthesised, the gold answer is negative — and `abs()` guarantees a miss. The
tolerance leaves no room to absorb it: `math.isclose(rel_tol=0.0, abs_tol=0.01)`.

Usage:
  PYTHONPATH=src python scripts/audit_sign.py --base aimed.zip
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trace_answer import logged_reads, read_grid  # noqa: E402


def negative_cell(text: str) -> bool:
    raw = str(text).strip()
    return bool(raw) and (raw.startswith("-") or (raw.startswith("(") and raw.endswith(")")))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    import pandas as pd

    tally: collections.Counter[str] = collections.Counter()
    offenders = []

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
        rows = payload if isinstance(payload, list) else (
            payload.get("predictions") or list(payload.values())[0])

        for row in rows:
            try:
                answer = float(row.get("answer"))
            except (TypeError, ValueError):
                tally["đáp án không phải số"] += 1
                continue

            frames = {}
            for item in row.get("evidence") or []:
                try:
                    grid = read_grid(archive, item["csv_path"])
                except KeyError:
                    continue
                if grid:
                    frames[item["variable"]] = pd.DataFrame(grid[1:], columns=grid[0])
            if not frames:
                tally["không có khung"] += 1
                continue
            reads = logged_reads(row.get("pandas_query") or "", frames)
            if not reads:
                tally["không đọc qua num()"] += 1
                continue
            # Only a single-cell answer is decidable: with several operands the sign
            # of the result depends on arithmetic this cannot see.
            unique = {(v, r, c): cell for v, r, c, cell in reads}
            if len(unique) != 1:
                tally["nhiều ô"] += 1
                continue
            cell = next(iter(unique.values()))
            if not negative_cell(cell):
                tally["ô dương"] += 1
                continue
            if answer < 0:
                tally["ô âm, đáp án âm — đúng"] += 1
            else:
                tally["Ô ÂM NHƯNG ĐÁP ÁN DƯƠNG"] += 1
                offenders.append((row["id"], answer, cell, row["question"][:74]))

    total = sum(tally.values())
    print(f"{args.base}: {total} câu\n")
    for name, count in tally.most_common():
        print(f"  {name:30s} {count:5d}  {count / max(total, 1):5.1%}")
    print()
    for qid, answer, cell, question in offenders[: args.show]:
        print(f"  id={qid:<5d} nộp={answer!s:<18s} ô={cell!r}")
        print(f"        {question}")


if __name__ == "__main__":
    main()
