"""What shape are the SFT targets, and can they be scored without a table store?

Decides how to score the held-out A/B. If nearly every target is a single
`result = num(dfK, r, c)` then a parsed (frame, row, col) tuple is a faithful
grade and needs nothing but the jsonl. Anything more varied has to be executed.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SINGLE_NUM = re.compile(r"^result = num\(df(\d*), (\d+), (\d+)\)$")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [
        json.loads(line)
        for line in (ROOT / "artifacts" / "sft_easy.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    shapes: Counter[str] = Counter()
    others: list[str] = []
    for row in rows:
        target = row["messages"][-1]["content"].strip()
        if SINGLE_NUM.match(target):
            shapes["single num()"] += 1
        elif target.count("\n") == 0:
            shapes["other one-liner"] += 1
            others.append(target)
        else:
            shapes[f"{target.count(chr(10)) + 1} lines"] += 1
            others.append(target.replace("\n", " ; "))

    print(f"{len(rows)} targets")
    for shape, count in shapes.most_common():
        print(f"  {count:>4}  {shape}")

    if others:
        print("\nnon-single-num samples:")
        for target in others[:12]:
            print(f"  {target[:120]}")

    # The eval slice is rows[:split] in train_qlora.py, so name it exactly.
    split = max(1, int(len(rows) * 0.15))
    print(f"\nheld-out slice = first {split} rows of the file")


if __name__ == "__main__":
    main()
