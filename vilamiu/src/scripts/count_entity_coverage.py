"""How many of the companies a question names does its program actually read?

Hand-tracing ten shipped answers found three wrong, and two failed the same way:
a question naming three companies was answered from two frames, and a screen over
four was answered from one company's revenue. The cells that were read were right.
The missing ones are what lost the question.

That is mechanically checkable. `parse_question` says how many tickers a question
names; running the program under a tracer says how many distinct frames it read a
number from. When the second is smaller than the first the answer cannot be right,
whatever the cells contain — no gold answer needed.

Usage:
  PYTHONPATH=src python scripts/count_entity_coverage.py --base direct14b_repair.zip
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
sys.path.insert(0, str(ROOT / "scripts"))

from trace_answer import logged_reads, read_grid  # noqa: E402

from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402


# Phrases that make every named company an operand: a sum, an average, a ranking,
# a count, or a screen over the group.
COHORT_RE = re.compile(
    r"xét nhóm|trong nhóm|xét các|các doanh nghiệp|các công ty|nhóm mã|"
    r"có bao nhiêu (doanh nghiệp|công ty|ngân hàng)|"
    r"tổng (giá trị |số |cộng )?[^,?]{0,60}của [^?]{0,40}, |"
    r"trung bình|trung vị|cao nhất trong|thấp nhất trong|lớn nhất trong|"
    r"chênh lệch [^?]{0,50} giữa [^?]{0,40} và ", re.I)


def ticker_of(csv_path: str) -> str:
    """The ticker a bound csv belongs to, from its file name."""

    name = csv_path.split("/")[-1]
    return name.split("_")[0].upper()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    import pandas as pd

    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    tally: collections.Counter[str] = collections.Counter()
    short: list[tuple] = []

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
        rows = payload if isinstance(payload, list) else (
            payload.get("predictions") or list(payload.values())[0])

        for row in rows:
            parsed = parse_question(0, row["question"], roster)
            named = len(set(parsed.tickers))
            if named < 2:
                tally["câu một công ty"] += 1
                continue
            # A related-party question names two companies and needs one frame:
            # "Vay dài hạn với Công ty X của công ty mẹ Y" is a single cell in Y's
            # report about X. Counting those as under-read inflated the class by a
            # third. Only genuine group aggregations and screens require one read
            # per company.
            if not COHORT_RE.search(row["question"]):
                tally["nêu nhiều tên, không phải câu nhóm"] += 1
                continue
            frames, owner = {}, {}
            for item in row.get("evidence") or []:
                try:
                    grid = read_grid(archive, item["csv_path"])
                except KeyError:
                    continue
                if grid:
                    frames[item["variable"]] = pd.DataFrame(grid[1:], columns=grid[0])
                    owner[item["variable"]] = ticker_of(item["csv_path"])
            if not frames:
                tally["không có khung nào"] += 1
                continue
            reads = logged_reads(row.get("pandas_query") or "", frames)
            touched = {owner.get(variable) for variable, _, _, _ in reads}
            touched.discard(None)
            tally["đủ công ty"] += 1 if len(touched) >= named else 0
            if len(touched) < named:
                tally["THIẾU công ty"] += 1
                short.append((row["id"], named, len(touched), row["question"][:88]))

    total = sum(tally.values())
    print(f"{args.base}: {total} câu\n")
    for name, count in tally.most_common():
        print(f"  {name:24s} {count:5d}")
    multi = tally["đủ công ty"] + tally["THIẾU công ty"]
    if multi:
        print(f"\ntrong {multi} câu nhiều công ty, THIẾU ở "
              f"{tally['THIẾU công ty']} câu ({tally['THIẾU công ty'] / multi:.1%})")
    for qid, named, touched, question in short[: args.show]:
        print(f"  id={qid:<5d} nêu {named} công ty, đọc {touched}: {question}")


if __name__ == "__main__":
    main()
