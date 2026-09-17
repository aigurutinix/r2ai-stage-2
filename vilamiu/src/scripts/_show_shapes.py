"""One generated record per shape, in full, for reading rather than counting.

A generated pair can verify perfectly and still be useless: if the Vietnamese
reads unlike the exam, the model learns our phrasing instead of the skill. That
failure is invisible to every automatic check, so the sample has to be read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "artifacts/_shapes_smoke.jsonl")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    seen: set[str] = set()
    for row in rows:
        shape = row["shape"]
        if shape in seen:
            continue
        seen.add(shape)
        print("=" * 78)
        print(f"[{shape}]  tables cited: {len(row['relevant_tables'])}")
        print(f"Q: {row['question']}")
        print(f"A: {row['answer']}")
        print("program:")
        for line in row["pandas_query"].splitlines():
            print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
