"""Split the exam into the blocks that decide the score, and say how big each is.

Every mechanism here was aimed at "a question" as if the 1012 were alike. They are not,
and the shape of a question fixes what any reader can possibly do with it:

  437  one money figure, one year, no screening   one cell — the block that decides the
                                                  score, and the only one where a good
                                                  reader converts directly into points
   54  money across years                         two cells
   94  a ratio inside one year                    two cells and a formula
   59  a ratio across years                       three or four cells
   52  which year was highest                     k cells, but only their ORDER matters,
                                                  so unit and scale errors cancel
  295  screening across companies                 fifteen to thirty cells, every one of
                                                  them right — unreachable at any read
                                                  accuracy this pipeline has shown

A single-cell reader shipped onto a multi-year or screening question is wrong by
construction, so these lists exist to keep it off them.

Usage:
  python scripts/fresh/blocks.py --out artifacts/fresh/blocks.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MONEY_RE = re.compile(
    r"bao nhiêu (đồng|nghìn đồng|triệu đồng|tỷ đồng|trăm tỷ đồng|nghìn tỷ đồng)", re.I)
PCT_RE = re.compile(r"phần trăm|%|tỷ lệ|tỉ lệ", re.I)
TIMES_RE = re.compile(r"bao nhiêu lần", re.I)
WHICH_YEAR_RE = re.compile(r"năm nào", re.I)
COUNT_RE = re.compile(
    r"bao nhiêu (công ty|đơn vị|doanh nghiệp|khoản mục|thành viên|chi nhánh|"
    r"cổ đông|người|nhân viên|lao động)", re.I)
# Wording that makes the answer depend on comparing several companies or periods.
SCREEN_RE = re.compile(
    r"trung vị|cao nhất|thấp nhất|lớn nhất|nhỏ nhất|trong nhóm|trong số|xét các|"
    r"các doanh nghiệp có mã|đồng thời|bao nhiêu doanh nghiệp|bao nhiêu công ty", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# An average or a total over several companies needs one cell per company. Neither word
# appears in the screening vocabulary, so two questions of this shape reached the
# single-cell list — an average of four companies' finance income and an average of three
# banks' provisions — where a one-cell answer is wrong however well it is read.
AGGREGATE_RE = re.compile(r"trung bình|bình quân|tổng .{0,40}(và|,)", re.I)


def block_of(text: str, companies: int = 1) -> str:
    # More than one company named means more than one cell, whatever else the wording
    # looks like. This is the only signal that catches a plain list of tickers.
    if companies > 1:
        return "nhieu cong ty"
    if AGGREGATE_RE.search(text):
        return "tong hop nhieu o"
    if COUNT_RE.search(text):
        return "dem cong ty"
    if WHICH_YEAR_RE.search(text):
        return "nam nao"
    screening = bool(SCREEN_RE.search(text))
    multi_year = len(set(YEAR_RE.findall(text))) > 1
    if TIMES_RE.search(text):
        return "so lan"
    if MONEY_RE.search(text):
        if screening:
            return "tien — sang loc"
        return "tien — nhieu nam" if multi_year else "tien — MOT O"
    if PCT_RE.search(text):
        if screening:
            return "ty le — sang loc"
        return "ty le — nhieu nam" if multi_year else "ty le — mot nam"
    if screening:
        return "khac — sang loc"
    return "khac — nhieu nam" if multi_year else "khac — don gian"


# The blocks a single-cell reader may be shipped onto.
SINGLE_CELL = ("tien — MOT O", "khac — don gian")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/fresh/blocks.json")
    args = parser.parse_args()

    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from resolve_ticker import TickerResolver
    resolver = TickerResolver()

    groups: dict[str, list[int]] = {}
    for question in questions:
        found = resolver.resolve(question["question"])
        groups.setdefault(
            block_of(question["question"], len(found)), []).append(question["id"])

    (ROOT / args.out).write_text(
        json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    counts = Counter({name: len(ids) for name, ids in groups.items()})
    for name, count in counts.most_common():
        mark = " <- mot o" if name in SINGLE_CELL else ""
        print(f"  {count:5d}  {name}{mark}")
    single = sum(len(groups.get(name, [])) for name in SINGLE_CELL)
    print(f"\nkhoi mot-o: {single} cau ({100 * single / len(questions):.0f}%)")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
