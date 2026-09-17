"""Why does a question get no canonical block? One reason per question, counted.

The model run covers 432 of 1012 questions, and that ceiling is the index's, not the
model's: a block is rendered per (company, year, scope), so a question the block
builder cannot place gets no prompt at all. Three refusals do all the work, and they
need three unrelated fixes:

  several companies    the block is one company's statements, so a cohort question has
                       no single block. Fixable by rendering two to four compact
                       blocks, which needs the note index dropped to fit the window.
  no ticker            the company is written a way `code_stock.csv` does not match.
  no statements        the primary statements for that company-year-scope never
                       parsed. Mostly credit institutions — only 8 of 40 bank reports
                       classify at all — plus reports whose unit line is missing.

Usage:  python scripts/fresh/why_no_block.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from render_block import Corpus  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    corpus = Corpus(ROOT / args.index, ROOT / args.notes)
    resolver = TickerResolver()
    banks = set()
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record["bank"]:
                banks.add(record["ticker"])
    # Every ticker flagged a credit institution, whether or not its statements parsed.
    import csv as csv_mod
    rows = list(csv_mod.DictReader(
        (ROOT / "data" / "file_filter.csv").open(encoding="utf-8-sig")))
    ticker_col = next(c for c in rows[0] if "CK" in c)
    level2 = next(c for c in rows[0] if "cấp 2" in c)
    all_banks = {r[ticker_col] for r in rows if "tín dụng" in (r[level2] or "")}

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for question in questions:
        text = question["question"]
        found = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        if not found:
            bucket = "khong nhan ra ma cong ty nao"
        elif len(found) > 1:
            bucket = f"nhieu cong ty ({len(found)})"
        elif not years:
            bucket = "khong nhan ra nam nao"
        else:
            ticker = next(iter(found))
            scope = "separate" if PARENT_RE.search(text) else "consolidated"
            year = max(years)
            block = corpus.block(ticker, year, scope) or corpus.block(
                ticker, year, "consolidated" if scope == "separate" else "separate")
            if block:
                bucket = "CO KHOI"
            elif ticker in all_banks:
                bucket = "khong co bao cao parse duoc — TO CHUC TIN DUNG"
            else:
                bucket = "khong co bao cao parse duoc — doanh nghiep"
        counters[bucket] += 1
        examples.setdefault(bucket, []).append(
            f"    id={question['id']:<5d} {text[:96]}")

    total = sum(counters.values())
    print(f"{total} cau\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    grouped = sum(v for k, v in counters.items() if k.startswith("nhieu cong ty"))
    print(f"\n  (tong 'nhieu cong ty': {grouped})")
    for name in sorted(counters):
        if name == "CO KHOI":
            continue
        print(f"\n--- {name}")
        for line in examples[name][:args.show]:
            print(line)


if __name__ == "__main__":
    main()
