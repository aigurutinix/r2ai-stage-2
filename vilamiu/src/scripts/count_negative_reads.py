"""How many shipped answers rest on a failed row lookup that read silently anyway?

`find_row` returns -1 when no label matches, and the prompt tells the model to
check for that. Many programs do not. `num(frame, -1, c)` then reads
`frame.iloc[-1, c]` — the LAST row of the table — without complaint. In a balance
sheet the last row is the total, so a failed lookup for "tài sản cố định vô hình"
comes back as total assets, and a share-of-total question divides the total by
itself and reports exactly 100.0.

Twelve answers in the 655-700 block are exactly 100.0 for that reason. This counts
the whole class: every answer whose program read a cell at a negative row index.

Usage:
  PYTHONPATH=src python scripts/count_negative_reads.py --base direct14b_repair.zip
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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    import pandas as pd

    tally: collections.Counter[str] = collections.Counter()
    hits = []
    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
        rows = payload if isinstance(payload, list) else (
            payload.get("predictions") or list(payload.values())[0])
        for row in rows:
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
            negative = [read for read in reads if read[1] < 0]
            if negative:
                tally["ĐỌC CHỈ SỐ ÂM"] += 1
                hits.append((row["id"], row.get("answer"), len(negative),
                             row["question"][:82]))
            else:
                tally["đọc bình thường"] += 1

    total = sum(tally.values())
    print(f"{args.base}: {total} câu\n")
    for name, count in tally.most_common():
        print(f"  {name:24s} {count:5d}  {count / max(total, 1):5.1%}")
    print()
    for qid, answer, count, question in hits[: args.show]:
        print(f"  id={qid:<5d} nộp={answer!s:<12s} ({count} ô âm)  {question}")


if __name__ == "__main__":
    main()
