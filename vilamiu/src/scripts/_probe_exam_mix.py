"""How many exam questions are multi-company sums — the thing medium teaches?

The derived pool costs real work to salvage: 30% of its records name a company
they cannot answer for, 59% share an answer with another question, and the gold
programs read anywhere from 1 to 12 cells. Before building a multi-cell target
format for it, this measures the prize: the share of the 1,012 exam questions
that actually name more than one company, and the share that name more than one
year. Those are the two shapes a single-cell target cannot express.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402

SUM_WORD = re.compile(r"\btổng\b", re.I)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    path = ROOT / "data" / "questions" / "questions.jsonl"

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"{len(rows)} exam questions\n")

    tickers: collections.Counter[int] = collections.Counter()
    years: collections.Counter[int] = collections.Counter()
    multi_company_sum = 0
    examples: list[str] = []

    for i, row in enumerate(rows):
        text = row.get("question") or row.get("text") or ""
        parsed = parse_question(i, text, roster)
        n_tickers = len(parsed.tickers)
        tickers[n_tickers] += 1
        years[len(parsed.years)] += 1
        if n_tickers > 1:
            multi_company_sum += 1
            if len(examples) < 5:
                examples.append(text)

    print("companies named per question:")
    for n, count in sorted(tickers.items()):
        print(f"  {n:>2} companies  {count:>5}  {count / len(rows):6.1%}")
    print("\nyears named per question:")
    for n, count in sorted(years.items()):
        print(f"  {n:>2} years      {count:>5}  {count / len(rows):6.1%}")

    print(f"\nmulti-company questions: {multi_company_sum}/{len(rows)} "
          f"= {multi_company_sum / len(rows):.1%}")
    for text in examples:
        print(f"  {text[:120]}")


if __name__ == "__main__":
    main()
