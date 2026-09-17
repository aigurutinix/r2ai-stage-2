"""Of the exam's multi-company questions, how many are the plain sum medium teaches?

The derived pool's shape is "tổng X của A và B" — add one cell per company. The
exam's multi-company questions look different at a glance: cohort screens, counts,
superlatives over a group. If the plain-sum share is small, salvaging the derived
pool buys little regardless of how well the salvage works, and the effort belongs
on the shapes that dominate.

Reuses the class regexes from `_class_sizes.py` so this cannot disagree with the
classification the rest of the analysis is built on.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _class_sizes import COHORT, RATIO_UNIT, SUPER  # noqa: E402

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402

COUNT = re.compile(r"bao nhiêu (?:doanh nghiệp|công ty|mã)", re.I)
SUM = re.compile(r"\btổng\b", re.I)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "questions" / "questions.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    buckets: collections.Counter[str] = collections.Counter()
    plain_sums: list[str] = []

    multi = 0
    for i, row in enumerate(rows):
        text = row["question"]
        parsed = parse_question(i, text, roster)
        if len(parsed.tickers) < 2:
            continue
        multi += 1
        if COUNT.search(text):
            name = "count over a group"
        elif COHORT.search(text):
            name = "cohort screen / rank"
        elif SUPER.search(text):
            name = "superlative"
        elif RATIO_UNIT.search(text):
            name = "ratio / share"
        elif SUM.search(text):
            name = "plain sum across companies"
            plain_sums.append(text)
        else:
            name = "other multi-company"
        buckets[name] += 1

    print(f"{multi} multi-company exam questions\n")
    for name, count in buckets.most_common():
        print(f"  {name:<30} {count:>4}  {count / multi:6.1%} of multi"
              f"  {count / len(rows):6.1%} of exam")

    print(f"\ntwo-company arithmetic: "
          f"{len(plain_sums)}/{len(rows)} = {len(plain_sums) / len(rows):.1%} of the exam")

    # The operation matters as much as the arity. The derived pool asks for
    # "tổng ... của A và B"; if the exam asks for "chênh lệch ... giữa A và B"
    # then training on that pool teaches the wrong operation on the same shape.
    difference = re.compile(r"chênh lệch|trừ đi|so với|nhiều hơn|ít hơn", re.I)
    diffs = [t for t in plain_sums if difference.search(t)]
    print(f"  of which differences (chênh lệch / trừ đi): {len(diffs)} "
          f"= {len(diffs) / max(1, len(plain_sums)):.0%}")
    print(f"  actual additions:                          "
          f"{len(plain_sums) - len(diffs)}")
    for text in plain_sums[:5]:
        print(f"  {text[:115]}")


if __name__ == "__main__":
    main()
