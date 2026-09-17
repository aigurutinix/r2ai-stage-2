"""Estimate the tier mix of the 1,012 real questions.

`questions.jsonl` carries only `id` and `question` — the organisers did not
release the difficulty labels — so the split has to be inferred. It is worth
inferring because it decides how to weight a training set: the `easy` tier
generates at ~240 records/hour with no GPU, while `medium` managed 2.7/hour and
needs one, so knowing what share of the exam each covers is the difference
between a plan that fits in a night and one that does not.

The classification uses the same signals the organisers' own generator branches
on — how many companies, how many periods, whether a ratio is named, whether the
question screens a group — rather than a new heuristic invented here.

Usage:  PYTHONPATH=src python scripts/_probe_tier_mix.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import compose, lookup as lookup_mod, ratio as ratio_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402


def main() -> None:
    questions = parse_all(
        ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")

    tiers: Counter[str] = Counter()
    for question in questions:
        tickers = max(1, len(question.tickers))
        years = max(1, len(question.years))
        is_ratio = (ratio_mod.shape(question) is not None
                    or ratio_mod.compound_shape(question) is not None)
        is_screen = (compose.screen_shape(question) is not None
                     or compose.SCREEN_RE.search(question.question) is not None)

        if is_screen or (tickers > 1 and years > 1):
            tiers["hard / cohort-screen"] += 1
        elif is_ratio:
            tiers["intermediate / ratio"] += 1
        elif tickers > 1 or years > 1:
            tiers["medium / multi-operand"] += 1
        elif lookup_mod.is_single_lookup(question.question):
            tiers["easy / single cell"] += 1
        else:
            tiers["easy-ish / one table"] += 1

    total = len(questions)
    print(f"{total} questions\n")
    for name, count in tiers.most_common():
        print(f"  {name:26s} {count:4d}   {count / total:5.1%}")

    single = tiers["easy / single cell"] + tiers["easy-ish / one table"]
    print(f"\n  reachable by easy-tier supervision : {single:4d}   "
          f"{single / total:5.1%}")
    print(f"  needs derived supervision          : {total - single:4d}   "
          f"{(total - single) / total:5.1%}")
    print("\n  The first number is the ceiling on what a training set built only")
    print("  from the easy tier can address. The second is what it cannot, and")
    print("  that tier costs a GPU and ran ~90x slower.")


if __name__ == "__main__":
    main()
