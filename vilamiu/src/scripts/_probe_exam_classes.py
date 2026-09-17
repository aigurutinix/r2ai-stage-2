"""What share of the exam is the single-cell lookup our SFT set is made of?

The previous fine-tune scored 69.4% on held-out and went backwards on the exam.
Two explanations have been floated — no blind pairs, wrong distractors — and both
are about *how* the pairs were built. This checks a third that is about *what*
they contain: every pair in the set is a one-cell lookup, so if the exam is
substantially not that, the set has a ceiling no amount of blind pairs will lift.

Classes are the ones already used in `_class_sizes.py`, applied to all 1,012
questions rather than only the multi-company slice.
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

from _class_sizes import COHORT, MEDIAN, MULTIYEAR, RATIO_UNIT, SUPER  # noqa: E402

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402

COUNT = re.compile(r"bao nhiêu (?:doanh nghiệp|công ty|mã)", re.I)
DIFFERENCE = re.compile(r"chênh lệch|trừ đi|nhiều hơn|ít hơn|tăng trưởng|so với năm", re.I)


def classify(text: str, tickers: int, years: int) -> str:
    if COUNT.search(text):
        return "count over a group"
    if COHORT.search(text) or tickers >= 3:
        return "cohort screen / rank"
    if MEDIAN.search(text):
        return "median filter"
    if SUPER.search(text):
        return "superlative"
    if RATIO_UNIT.search(text):
        return "ratio / percentage"
    if DIFFERENCE.search(text) or MULTIYEAR.search(text) or years >= 2:
        return "two-cell arithmetic"
    if tickers >= 2:
        return "two-company arithmetic"
    return "single-cell lookup"


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
    for i, row in enumerate(rows):
        text = row["question"]
        parsed = parse_question(i, text, roster)
        buckets[classify(text, len(parsed.tickers), len(parsed.years))] += 1

    total = len(rows)
    print(f"{total} exam questions\n")
    print(f"{'class':<26}{'n':>6}{'share':>9}")
    for name, count in buckets.most_common():
        print(f"{name:<26}{count:>6}{count / total:>9.1%}")

    # The same classifier on the training set. A held-out score is only a
    # prediction of the exam to the extent the two distributions agree, and this
    # is the comparison that says whether 69.4% held-out meant anything.
    sft_path = ROOT / "artifacts" / "sft_locate2.jsonl"
    if not sft_path.exists():
        return
    sft: collections.Counter[str] = collections.Counter()
    for line in sft_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)["meta"]
        text = meta.get("question") or ""
        parsed = parse_question(0, text, roster)
        sft[classify(text, len(parsed.tickers), len(parsed.years))] += 1
    n_sft = sum(sft.values())

    print(f"\n{'class':<26}{'exam':>9}{'training set':>14}{'gap':>9}")
    for name in [n for n, _ in buckets.most_common()]:
        exam_share = buckets[name] / total
        sft_share = sft[name] / n_sft if n_sft else 0.0
        print(f"{name:<26}{exam_share:>9.1%}{sft_share:>14.1%}"
              f"{sft_share - exam_share:>+9.1%}")

    single = buckets["single-cell lookup"]
    print(f"\nexam that is a one-cell lookup: {single}/{total} = {single / total:.1%}")
    print(f"training set that is:            "
          f"{sft['single-cell lookup']}/{n_sft} = "
          f"{sft['single-cell lookup'] / n_sft:.1%}")


if __name__ == "__main__":
    main()
