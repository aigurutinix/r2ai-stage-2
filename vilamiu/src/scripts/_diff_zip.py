"""Compare two submission zips: answers, programs and declared tables.

The repo has no git history, so a zip is the only durable record of what a
configuration produced. When the code that built the best submission is no longer
the code on disk, this is the instrument that says so — and, after a restoration,
the one that certifies the restoration is faithful (0 differing answers).

Usage:  python scripts/_diff_zip.py a.zip b.zip
"""

from __future__ import annotations

import json
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict[int, dict]:
    path = ROOT / "submissions" / name
    with zipfile.ZipFile(path) as z:
        return {r["id"]: r for r in json.loads(z.read("submission.json"))}


def close(a, b) -> bool:
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if x == y:
        return True
    scale = max(abs(x), abs(y))
    return scale > 0 and abs(x - y) / scale <= 2e-4  # the scorer's tolerance


def main() -> None:
    left_name, right_name = sys.argv[1], sys.argv[2]
    left, right = load(left_name), load(right_name)
    ids = sorted(set(left) | set(right))

    same = diff = 0
    for i in ids:
        if i not in left or i not in right:
            diff += 1
            continue
        if close(left[i].get("answer"), right[i].get("answer")):
            same += 1
        else:
            diff += 1

    print(f"{left_name}  vs  {right_name}")
    print(f"  questions: {len(ids)}   same answer: {same}   DIFFERENT: {diff}")

    for name, recs in ((left_name, left), (right_name, right)):
        counts = [len(r["relevant_tables"]) for r in recs.values()]
        zeros = sum(
            1 for r in recs.values()
            if (r.get("pandas_query") or "").strip() in ("", "result = 0.0"))
        print(f"\n  {name}")
        print(f"    declared refs: mean {statistics.mean(counts):.2f}  "
              f"histogram {dict(sorted(Counter(counts).items()))}")
        print(f"    programs that are `result = 0.0`: {zeros}")
        print(f"    distinct programs: {len({r.get('pandas_query') for r in recs.values()})}")

    if diff:
        print(f"\n  first 15 differing ids:")
        shown = 0
        for i in ids:
            if i in left and i in right and close(
                    left[i].get("answer"), right[i].get("answer")):
                continue
            print(f"    id={i:4d}  {left[i].get('answer')!r:>22}  ->  "
                  f"{right[i].get('answer')!r}")
            shown += 1
            if shown >= 15:
                break


if __name__ == "__main__":
    main()
