"""Identify which branch wrote each shipped program, from the program's own shape.

The patch that just scored measured something we could not measure before: on the
68 provably-wrong questions it replaced, EXECUTION moved 0.3281 -> 0.3379, which
is ~5 of the ~34 that fall in the graded half — a hit rate near **15%** for the
14B on the hardest pool in the set.

That number is only actionable if it can be compared against the branches it
would displace, and those are known from the leaderboard history:

    label matcher   42.8%     <- do not touch
    llm (8B)        27%
    locate          13.3%     <- comparable, leave alone
    plan             7.2%     <- candidate
    best-effort      5.9%     <- candidate

So the next question is purely one of identification: which of the 1,012 shipped
programs came from `plan` and which from the best-effort fallback. The zip records
no branch label, but each branch has a distinct code generator, so the shape of
the program is the label.

Usage:  PYTHONPATH=src python scripts/_probe_branch_shapes.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SUB = ROOT / "submissions" / "screen_ratio_gated.zip"


def main() -> None:
    with zipfile.ZipFile(SUB) as z:
        records = json.loads(z.read("submission.json"))

    shapes: Counter[str] = Counter()
    samples: dict[str, list[tuple[int, str]]] = {}
    for record in records:
        code = (record.get("pandas_query") or "").strip()
        frames = sorted(set(re.findall(r"\bdf\d*\b", code)))
        if code in ("", "result = 0.0"):
            key = "placeholder"
        elif "def num(" in code or "def find_row(" in code:
            key = "llm_prelude"
        elif len(frames) >= 2:
            key = "multi_frame"
        else:
            # One frame. Separate the generators by the constructs each emits.
            marks = []
            if ".astype(str)" in code:
                marks.append("astype")
            if "str.strip()" in code:
                marks.append("strip")
            if "hits.index[0]" in code or "labels[labels ==" in code:
                marks.append("labelmatch")
            if "abs(" in code:
                marks.append("abs")
            if re.search(r"iloc\[\s*\d+\s*,\s*\d+\s*\]", code):
                marks.append("fixed_cell")
            if "sum()" in code or "max()" in code or "min()" in code:
                marks.append("aggregate")
            key = "single:" + ("+".join(marks) if marks else "bare")
        shapes[key] += 1
        samples.setdefault(key, []).append((record["id"], code))

    print(f"{len(records)} shipped programs, grouped by shape\n")
    for key, count in shapes.most_common():
        print(f"  {count:5d}  {key}")

    print("\n--- one example per shape ---")
    for key, _ in shapes.most_common():
        qid, code = samples[key][0]
        print(f"\n### {key}  (id={qid})")
        for line in code.splitlines()[:9]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
