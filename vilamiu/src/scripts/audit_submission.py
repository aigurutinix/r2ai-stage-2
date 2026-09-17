"""Inventory the defects in a submission, sized, without needing gold answers.

Hand-reading found three real defects today and each was invisible to aggregate
measurement. But reading one answer at a time does not say how big a class is, and
a class worth fixing has to be worth more than the eight questions the last fix
moved. These checks are all things that are wrong *on their own terms* — no gold
required — so the count is the size of the class.

Each check is deliberately conservative: it only fires where the answer cannot be
right, not merely where it looks odd. A share above 100%, a count that is not a
whole number, a "năm nào" question answered with a figure.

Usage:
  PYTHONPATH=src python scripts/audit_submission.py --base planv3_clean.zip
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402

YEAR_Q = re.compile(r"(năm nào|vào năm nào|thời điểm nào|năm bao nhiêu)", re.I)
# Only a share *of a whole* is capped at 100. A growth rate is not: "tỷ lệ tăng
# doanh thu" at 105.61% and "tăng bao nhiêu %" at 111.61% are both legitimate, and
# a check that flags them reports 80 defects where about 30 exist.
SHARE_Q = re.compile(
    r"tỷ trọng|tỉ trọng|chiếm bao nhiêu (phần trăm|%)|tỷ lệ sở hữu|"
    r"tỷ lệ lợi ích|phần trăm sở hữu", re.I)
GROWTH_Q = re.compile(r"tăng|giảm|tăng trưởng|thay đổi|biến động|CAGR", re.I)
PERCENT_Q = re.compile(r"bao nhiêu (phần trăm|%)|\(%\)", re.I)
TIMES_Q = re.compile(r"bao nhiêu lần|gấp bao nhiêu", re.I)
COUNT_Q = re.compile(r"có bao nhiêu (doanh nghiệp|công ty|ngân hàng|đơn vị|mã)", re.I)
# Line items that are always billions of dong at a listed company.
BIG_METRIC = re.compile(
    r"tổng tài sản|tổng cộng tài sản|vốn chủ sở hữu|nợ phải trả|doanh thu thuần|"
    r"tổng nguồn vốn|cho vay khách hàng|tiền gửi của khách hàng", re.I)
UNIT_SCALE = (("nghìn tỷ", 1e12), ("tỷ", 1e9), ("triệu", 1e6), ("nghìn", 1e3))


def unit_of(question: str):
    lowered = question.lower()
    for name, scale in UNIT_SCALE:
        if re.search(rf"bao nhiêu[^?]*\b{re.escape(name)}\b", lowered) or \
           re.search(rf"\({re.escape(name)}\s*đồng\)", lowered):
            return name, scale
    return None, None


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--show", type=int, default=2, help="examples per class")
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    rows = payload if isinstance(payload, list) else (
        payload.get("predictions") or list(payload.values())[0])
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    counts: collections.Counter[str] = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)

    def flag(name: str, row, note: str = "") -> None:
        counts[name] += 1
        if len(examples[name]) < args.show:
            examples[name].append((row["question"][:96], row.get("answer"), note))

    for row in rows:
        question = row.get("question", "")
        answer = as_float(row.get("answer"))
        query = row.get("pandas_query") or ""
        counts["total"] += 1

        if answer is None:
            flag("no numeric answer", row)
            continue

        if YEAR_Q.search(question) and not (
                float(answer).is_integer() and 1990 <= answer <= 2100):
            flag("asks a year, answers a figure", row)

        if SHARE_Q.search(question) and not TIMES_Q.search(question) and \
                not GROWTH_Q.search(question) and (answer > 100 or answer < 0):
            flag("a share of a whole outside 0..100", row)
        elif PERCENT_Q.search(question) and not TIMES_Q.search(question) and \
                abs(answer) > 1e5:
            flag("a percentage above 100,000", row)

        if TIMES_Q.search(question) and (answer > 500 or answer < 0):
            flag("a multiple over 500x or negative", row)

        if COUNT_Q.search(question):
            group = len(parse_question(0, question, roster).tickers)
            if not float(answer).is_integer():
                flag("counts companies, answer not whole", row)
            elif group and answer > group:
                flag("counts more companies than named", row, f"named {group}")

        if answer == 0:
            flag("answer is exactly zero", row)

        name, scale = unit_of(question)
        if scale and BIG_METRIC.search(question) and abs(answer) * scale < 1e9:
            flag("headline figure under one billion dong", row, f"in {name}")

        if re.search(r"\bresult\s*=\s*-?[\d.]+\s*$", query, re.M) and \
                "num(" not in query and "iloc" not in query:
            flag("query asserts a constant", row)

    total = counts.pop("total", 0)
    print(f"{args.base}: {total} predictions\n")
    print(f"{'defect class':44s} {'count':>6s} {'share':>7s}")
    for name, count in counts.most_common():
        print(f"  {name:42s} {count:6d} {count / max(total, 1):7.1%}")
    print()
    for name, _ in counts.most_common():
        for question, answer, note in examples[name]:
            suffix = f"  ({note})" if note else ""
            print(f"  [{name}]{suffix}")
            print(f"     {question}")
            print(f"     -> {answer}")


if __name__ == "__main__":
    main()
