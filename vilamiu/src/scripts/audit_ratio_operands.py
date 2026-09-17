"""A ratio needs two numbers. Count the ratio answers computed from fewer.

"Tỷ trọng tài sản cố định hữu hình trên tổng tài sản" is a quotient: it cannot be
read out of one cell. Neither can "tỷ suất lợi nhuận", "vòng quay vốn chủ sở hữu",
or "gấp bao nhiêu lần". So a program that read a single cell and reported a ratio is
wrong by construction, and that is decidable without any gold answer — the same
argument that found 139 cohort questions reading too few companies.

389 of 1012 questions ask for something with no money unit, which is where these
live. The single-cell money answers were checked separately by
`audit_unit_math.py`; this covers the other pool.

Usage:
  PYTHONPATH=src python scripts/audit_ratio_operands.py --base aimed.zip
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

# Phrasings that name a quotient of two figures. "tăng trưởng" and "biến động" are
# included: a growth rate is (new - old) / old, which needs two cells as well.
RATIO_RE = re.compile(
    r"tỷ lệ|tỉ lệ|tỷ trọng|tỉ trọng|tỷ suất|tỷ số|hệ số|vòng quay|biên lợi nhuận|"
    r"gấp bao nhiêu|bao nhiêu lần|tăng trưởng|tốc độ tăng|biến động|"
    r"chiếm bao nhiêu", re.I)

# A ratio the statement prints directly, so one cell is legitimate: ownership and
# voting-rights tables store the percentage itself.
DIRECT_RE = re.compile(
    r"tỷ lệ sở hữu|tỷ lệ lợi ích|tỷ lệ quyền biểu quyết|phần trăm sở hữu|"
    r"tỷ lệ nắm giữ|lãi suất", re.I)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    import pandas as pd

    tally: collections.Counter[str] = collections.Counter()
    offenders = []

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
        rows = payload if isinstance(payload, list) else (
            payload.get("predictions") or list(payload.values())[0])

        for row in rows:
            question = row["question"]
            if not RATIO_RE.search(question):
                tally["không phải câu tỷ số"] += 1
                continue
            if DIRECT_RE.search(question):
                tally["tỷ lệ in sẵn trong bảng"] += 1
                continue

            frames = {}
            for item in row.get("evidence") or []:
                try:
                    grid = read_grid(archive, item["csv_path"])
                except KeyError:
                    continue
                if grid:
                    frames[item["variable"]] = pd.DataFrame(grid[1:], columns=grid[0])
            if not frames:
                tally["không có khung"] += 1
                continue

            reads = logged_reads(row.get("pandas_query") or "", frames)
            cells = {(variable, r, c) for variable, r, c, _ in reads}
            if not reads:
                tally["không đọc qua num()"] += 1
                continue
            if len(cells) >= 2:
                tally["đủ hai toán hạng"] += 1
                continue
            tally["MỘT Ô CHO MỘT TỶ SỐ"] += 1
            offenders.append((row["id"], row.get("answer"), question[:80]))

    total = sum(tally.values())
    print(f"{args.base}: {total} câu\n")
    for name, count in tally.most_common():
        print(f"  {name:26s} {count:5d}  {count / max(total, 1):5.1%}")
    scope = tally["đủ hai toán hạng"] + tally["MỘT Ô CHO MỘT TỶ SỐ"]
    if scope:
        share = tally["MỘT Ô CHO MỘT TỶ SỐ"] / scope
        print(f"\ntrong {scope} câu tỷ số truy được, {tally['MỘT Ô CHO MỘT TỶ SỐ']} "
              f"câu chỉ đọc một ô ({share:.1%})")
    for qid, answer, question in offenders[: args.show]:
        print(f"  id={qid:<5d} nộp={answer!s:<16s} {question}")


if __name__ == "__main__":
    main()
