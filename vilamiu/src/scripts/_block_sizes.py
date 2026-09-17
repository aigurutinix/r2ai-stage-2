"""Size the question blocks the current machinery is not built for.

The ratio funnel turned up 88 questions that the ratio branch cannot parse, and the
wording says why: most are not quotients at all. "Tỷ lệ sở hữu Công ty CP Gang thép
Hòa Phát của HPG" is a single cell — the ownership table has one row per subsidiary
and a column headed "Tỷ lệ sở hữu (%)". No division, no formula, just a row named
after a company and a column named after the concept.

The second block is differences: "biến động / chênh lệch / thay đổi giữa cuối năm
2019 và cuối năm 2023" needs one row at two columns, or two reports. The one-cell
reader answers those with a single period and is wrong by construction.

Both are measured here against what `aimed` currently answers, using only checks
that need no gold: an ownership share must sit in (0, 100]; a difference must not
equal either operand.

Usage:  PYTHONPATH=src python scripts/_block_sizes.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OWNERSHIP = re.compile(
    r"tỷ lệ\s+(?:sở hữu|lợi ích|quyền biểu quyết|biểu quyết|nắm giữ|góp vốn|vốn góp)",
    re.I)
DIFFERENCE = re.compile(
    r"\b(?:biến động|chênh lệch|thay đổi|mức tăng|mức giảm|tăng bao nhiêu|"
    r"giảm bao nhiêu|tăng thêm)\b", re.I)
TWO_PERIODS = re.compile(
    r"(?:giữa|từ)\s.{0,60}?\b(?:19|20)\d{2}\b.{0,40}?\b(?:và|đến|so với)\b.{0,40}?"
    r"\b(?:19|20)\d{2}\b", re.I | re.S)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with zipfile.ZipFile(ROOT / "submissions" / "aimed.zip") as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))

    own = [r for r in rows if OWNERSHIP.search(r["question"])]
    diff = [r for r in rows if DIFFERENCE.search(r["question"])
            and TWO_PERIODS.search(r["question"])]
    overlap = {r["id"] for r in own} & {r["id"] for r in diff}

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    in_band = 0
    for row in own:
        value = as_float(row.get("answer"))
        if value is not None and 0 < value <= 100:
            in_band += 1
    print(f"khoi TY LE SO HUU / BIEU QUYET: {len(own)} cau")
    print(f"  dap an cua aimed nam trong (0,100]: {in_band} "
          f"({100 * in_band / max(1, len(own)):.0f}%)")
    print("  5 vi du dap an hien tai:")
    for row in own[:5]:
        print(f"    id={row['id']:4d} answer={row.get('answer')}  {row['question'][:96]}")

    print(f"\nkhoi CHENH LECH HAI KY: {len(diff)} cau"
          f"  (trung khoi tren: {len(overlap)})")
    print("  5 vi du:")
    for row in diff[:5]:
        print(f"    id={row['id']:4d} answer={row.get('answer')}  {row['question'][:96]}")

    both = {r["id"] for r in own} | {r["id"] for r in diff}
    print(f"\ntong hai khoi (khong trung): {len(both)} cau = "
          f"{100 * len(both) / len(rows):.1f}% de bai")


if __name__ == "__main__":
    main()
