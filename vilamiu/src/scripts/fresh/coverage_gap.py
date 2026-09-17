"""Where exactly does coverage stop?

141 of 1012 questions are answered, at 52% for the Mã số path and 43% for the note
path. Both are positive, so the next question is what blocks the other 871 — and the
blocks have different causes needing different work:

  no address book        the company-year-scope has no parseable primary statement, so
                         there is nothing for a note total to tie to
  book but no notes      statements parsed, but no non-statement table in the report
                         has a column total landing on one of their lines
  notes but no row       a note opened and no row matched the question's words
  multi-company          more than one ticker, which neither path addresses
  unit not money         the answer is a %, a multiple or a share count

Counting them separately is the difference between "open more coverage" and knowing
which of five unrelated problems to work on.

Usage:  python scripts/fresh/coverage_gap.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    book: set[tuple[str, str, str]] = set()
    for line in (ROOT / "artifacts" / "fresh" / "statements2.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            scope = ("separate" if "separate" in record["scope"]
                     else "consolidated" if "consolidated" in record["scope"]
                     else record["scope"])
            book.add((record["ticker"], record["year"], scope))

    tied: dict[tuple[str, str, str], int] = Counter()
    for line in (ROOT / "artifacts" / "fresh" / "notes.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            note = json.loads(line)
            if note.get("paired"):
                tied[(note["ticker"], note["year"], note["scope"])] += 1

    answered = set()
    for name in ("answer_plan.jsonl", "note_plan.jsonl"):
        path = ROOT / "artifacts" / "fresh" / name
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    answered.add(json.loads(line)["id"])

    tickers = {}
    for line in (ROOT / "data" / "code_stock.csv").read_text(
            encoding="utf-8").splitlines()[1:]:
        if "," in line:
            code, name = line.split(",", 1)
            tickers[code.strip()] = name.strip().strip('"')
    by_length = sorted(tickers.items(), key=lambda item: -len(item[1]))

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        text = question["question"]
        if question["id"] in answered:
            counters["DA TRA LOI"] += 1
            continue
        found = {c for c in tickers if re.search(rf"\b{re.escape(c)}\b", text)}
        for code, name in by_length:
            if name and name.casefold() in text.casefold():
                found.add(code)
        years = YEAR_RE.findall(text)
        _unit_name, unit = unit_of(text)

        if len(found) > 1:
            bucket = f"nhieu ma ({len(found)} ma)"
        elif not found:
            bucket = "khong nhan ra ma nao"
        elif not years:
            bucket = "khong nhan ra nam nao"
        elif not unit:
            bucket = "don vi khong phai tien (%, lan, co phieu)"
        else:
            ticker = next(iter(found))
            scope = "separate" if PARENT_RE.search(text) else "consolidated"
            key = (ticker, max(years), scope)
            if key not in book:
                other = ("separate" if scope == "consolidated" else "consolidated")
                if (ticker, max(years), other) in book:
                    bucket = "co so dia chi nhung PHAM VI khac"
                else:
                    bucket = "KHONG co so dia chi cho ma-nam-pham vi"
            elif not tied.get(key):
                bucket = "co so dia chi nhung KHONG thuyet minh nao noi duoc"
            else:
                bucket = "co ca hai nhung khong khop duoc chi tieu"
        counters[bucket] += 1
        if len(examples[bucket]) < 3:
            examples[bucket].append(f"    id={question['id']:<5d} {text[:104]}")

    total = sum(counters.values())
    print(f"{total} cau\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    for name in counters:
        if name == "DA TRA LOI":
            continue
        print(f"\n--- {name}")
        for line in examples[name]:
            print(line)


if __name__ == "__main__":
    main()
