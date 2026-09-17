"""How often does the pipeline answer a question with the wrong SHAPE?

The line-choice problem is running near 23%, and two attacks on it have stalled. Before
spending another GPU pass on structured extraction, this sizes the part of the failure
that is not line choice at all: answering a question with the wrong kind of computation.

A wrong shape is a guaranteed miss, not a near miss, and it is detectable without gold
because the question says what it wants:

  a difference across two periods  "chênh lệch / biến động giữa 2022 và 2018" needs two
                                   cells subtracted; one cell can never be right
  a filter then a report           "năm mà tỷ lệ X cao nhất, thì lợi nhuận là bao nhiêu"
                                   asks for the profit, not the ratio
  a sum across companies           "tổng của A, B và C" needs three cells
  a count                          "có bao nhiêu doanh nghiệp" is an integer, not money

So this counts, for the shipped submission, how many answers were produced by a
single-cell program where the question demands something else. That number is an upper
bound on what better shape classification can win, and it is the strongest part of the
structured-extraction idea.

Usage:  python scripts/fresh/check_shape.py --zip submissions/fresh_v7.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_ticker import TickerResolver  # noqa: E402

YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# Two periods named with a comparison between them.
DIFF_RE = re.compile(
    r"(?:chênh lệch|biến động|thay đổi|tăng|giảm|so với)[^?]{0,80}?"
    r"\b(?:19|20)\d{2}\b[^?]{0,40}?\b(?:và|đến|so với|sang)\b[^?]{0,40}?"
    r"\b(?:19|20)\d{2}\b", re.I)
FILTER_RE = re.compile(
    r"n[ăa]m (?:m[àa]|c[óo]|n[àa]o)|t[ạa]i n[ăa]m|v[àa]o n[ăa]m|"
    r"cao nh[ấa]t|th[ấa]p nh[ấa]t|l[ớo]n nh[ấa]t|nh[ỏo] nh[ấa]t|trung v[ịi]|"
    r"x[ée]t c[áa]c n[ăa]m|giai [đd]o[ạa]n", re.I)
SUM_RE = re.compile(r"t[ổo]ng .{0,40}(?:v[àa]|,).{0,40}(?:v[àa]|,)", re.I)
COUNT_RE = re.compile(r"bao nhi[êe]u (?:doanh nghi[ệe]p|c[ôo]ng ty|m[ãa])", re.I)


def cells_read(query: str) -> int:
    """How many distinct cells the emitted program reads."""

    return len(set(re.findall(r"iloc\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", query or "")))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/fresh_v7.zip")
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / args.zip) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))
    resolver = TickerResolver()

    counters: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for row in rows:
        if not (row.get("evidence") or []):
            continue
        text = row["question"]
        reads = cells_read(row.get("pandas_query", ""))
        years = set(YEAR_RE.findall(text))
        companies = len(resolver.resolve(text))

        needed = None
        if COUNT_RE.search(text):
            needed = "dem doanh nghiep"
        elif FILTER_RE.search(text) and len(years) >= 2:
            needed = "loc theo nam roi bao chi tieu"
        elif DIFF_RE.search(text):
            needed = "hieu giua hai ky"
        elif companies >= 2 and SUM_RE.search(text):
            needed = "ghep nhieu cong ty"
        elif companies >= 2:
            needed = "nhieu cong ty (phep chua ro)"

        if needed is None:
            counters["hinh dang mot o — hop le"] += 1
            continue
        enough = reads >= 2
        key = f"{needed} — chuong trinh doc {reads} o"
        counters[key] += 1
        if not enough:
            counters["  => SAI HINH DANG (1 o cho bai can nhieu o)"] += 1
            examples.setdefault(needed, []).append(
                f"    id={row['id']:<5d} {text[:100]}")

    answered = sum(v for k, v in counters.items() if not k.startswith("  =>"))
    wrong = counters["  => SAI HINH DANG (1 o cho bai can nhieu o)"]
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    print(f"\ntren {answered} cau co tra loi: {wrong} cau SAI HINH DANG "
          f"({100 * wrong / max(1, answered):.0f}%)")
    print("day la chan tren cua thu ma phan loai 'viec can lam' co the thang.")
    for name, lines in examples.items():
        print(f"\n--- {name}")
        for line in lines[:args.show]:
            print(line)


if __name__ == "__main__":
    main()
