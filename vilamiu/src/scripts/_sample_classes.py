"""Real exam wordings per class, so generated questions imitate the exam not us.

A templated question set is only useful if its phrasing and its answer *shape*
match what the exam asks. Getting the shape wrong is the expensive mistake: a
superlative question that wants a company name teaches a different task than one
that wants that company's figure, and only the exam text says which it is.

Prints full questions, grouped, with no truncation.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _probe_exam_classes import classify  # noqa: E402

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402

WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "questions" / "questions.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for i, row in enumerate(rows):
        text = row["question"]
        parsed = parse_question(i, text, roster)
        grouped[classify(text, len(parsed.tickers), len(parsed.years))].append(text)

    order = ["two-cell arithmetic", "ratio / percentage", "superlative",
             "cohort screen / rank", "count over a group"]
    for name in order:
        texts = grouped.get(name, [])
        print("=" * 78)
        print(f"{name}  ({len(texts)} questions)")
        print("=" * 78)
        # Spread the sample across the file rather than taking the head, so one
        # generator's phrasing habit does not stand in for the whole class.
        step = max(1, len(texts) // WANT)
        for text in texts[::step][:WANT]:
            print(f"- {text}")
        print()


if __name__ == "__main__":
    main()
