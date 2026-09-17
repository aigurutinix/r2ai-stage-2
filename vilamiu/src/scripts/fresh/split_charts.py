"""Two charts of accounts are mixed together, and that is what breaks the weak identity.

The identity checks come out at 99.5% on the balance sheet and 76.5% on
`kqkd: 20 = 10 - 11`. The label/code cross-check then showed why: the one-digit
"corrections" cluster on `01 -> 31`, `11 -> 21`, `10 -> 30`, and on labels like
"Vốn cổ phần đã phát hành" sitting under code 411. Those are not OCR losing a digit.
They are a different chart of accounts.

Credit institutions report under a separate statutory layout: an income statement
that begins with "Thu nhập lãi và các khoản thu nhập tương tự" rather than
"Doanh thu bán hàng", with no "Doanh thu thuần" line at all. Enterprise identities
therefore cannot hold for a bank, and averaging the two hides both.

This segments every statement by the chart its own labels imply, then re-runs the
identities per segment. If the enterprise-only rate for `20 = 10 - 11` jumps, the
parse was never the problem and the population was.

Usage:  python scripts/fresh/split_charts.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_identities import IDENTITIES, REL_TOL  # noqa: E402
from refine_codes import fold  # noqa: E402

# Wording that only a credit institution's statements carry.
BANK_MARKERS = (
    "thu nhap lai", "chi phi lai", "tien gui cua khach hang",
    "cho vay khach hang", "du phong rui ro tin dung",
    "thu nhap lai thuan", "hoat dong dich vu",
)
# Wording that only an enterprise's statements carry.
FIRM_MARKERS = (
    "doanh thu ban hang", "doanh thu thuan", "gia von hang ban",
    "loi nhuan gop", "hang ton kho",
)


def chart_of(record: dict) -> str:
    text = " ".join(fold(cell[1]) for cell in record["current"].values())
    bank = sum(1 for marker in BANK_MARKERS if marker in text)
    firm = sum(1 for marker in FIRM_MARKERS if marker in text)
    if bank and bank > firm:
        return "tin dung"
    if firm:
        return "doanh nghiep"
    return "khong ro"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = [json.loads(line) for line in
               (ROOT / "artifacts" / "fresh" / "statements.jsonl").read_text(
                   encoding="utf-8").splitlines() if line.strip()]

    charts = Counter()
    for record in records:
        record["_chart"] = chart_of(record)
        charts[(record["kind"], record["_chart"])] += 1
    print("phan bo bang theo bieu tai khoan:")
    for (kind, chart), count in sorted(charts.items()):
        print(f"  {kind:5s} {chart:13s} {count}")

    print("\ndang thuc, tach theo bieu tai khoan:")
    header = f"  {'dang thuc':40s} {'bieu':13s} {'du ma':>7s} {'DUNG':>7s} {'ty le':>7s}"
    print(header)
    for kind, target, plus, minus in IDENTITIES:
        rows: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            if record["kind"] != kind:
                continue
            for period in ("current", "prior"):
                cells = record[period]
                if not all(code in cells for code in (target,) + plus + minus):
                    continue
                expected = cells[target][0]
                total = sum(cells[c][0] for c in plus) - sum(cells[c][0] for c in minus)
                counter = rows[record["_chart"]]
                counter["du"] += 1
                if abs(expected - total) <= REL_TOL * max(abs(expected), abs(total), 1.0):
                    counter["dung"] += 1
        name = f"{kind}: {target} = " + " + ".join(plus) + (
            "" if not minus else " - " + " - ".join(minus))
        for chart in ("doanh nghiep", "tin dung", "khong ro"):
            counter = rows.get(chart)
            if not counter or not counter["du"]:
                continue
            print(f"  {name:40s} {chart:13s} {counter['du']:7d} {counter['dung']:7d} "
                  f"{100 * counter['dung'] / counter['du']:6.1f}%")


if __name__ == "__main__":
    main()
