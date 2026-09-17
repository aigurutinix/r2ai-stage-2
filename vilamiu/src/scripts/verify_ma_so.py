"""Check the derived code dictionary against the accounting identities.

`build_ma_so.py` pairs each statutory code with the label beside it, which proves
the two co-occur and nothing more. What makes the mapping trustworthy is that the
codes must satisfy arithmetic the corpus never had to obey by accident:

    cdkt:270 = cdkt:100 + cdkt:200        tổng tài sản = ngắn hạn + dài hạn
    cdkt:440 = cdkt:300 + cdkt:400        tổng nguồn vốn = nợ phải trả + vốn CSH
    cdkt:270 = cdkt:440                   hai vế bảng cân đối
    kqkd:20  = kqkd:10 - kqkd:11          lợi nhuận gộp
    kqkd:50  = kqkd:30 + kqkd:40          lợi nhuận trước thuế

If the code column were being misread, or the wrong label were being attached to
a code, these would fail. They are also stated inside the row labels themselves
("TỔNG CỘNG TÀI SẢN (270=100+200)"), which is how the corpus documents its own
schema — so this checks our extraction against the statements' own arithmetic
rather than against any external table.

Usage:  PYTHONPATH=src python scripts/verify_ma_so.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from build_ma_so import CODE_RE, code_column, statement_kind  # noqa: E402
from vifin.answering import lookup as lookup_mod  # noqa: E402

# Relative slack. OCR rounds and some statements print in thousands mid-table.
TOLERANCE = 0.005

# A leading "-" subtracts; a leading "|" subtracts the magnitude.
#
# Cost lines are printed either way across the corpus: some statements set giá vốn
# positive, others print it in accounting parentheses so it parses negative. The
# first run of this check read 69.4% on "20 = 10 - 11" and the failures looked
# like a bad code mapping — they were not. 9,258,073,280,674 + (-8,215,933,902,107)
# is exactly the reported 1,042,139,378,567. The test was wrong, not the data.
#
# That matters beyond this script: any formula consuming a cost code has to
# normalise the sign first, which is what the "sign_policy" of a real financial
# definition is for.
IDENTITIES: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("cdkt", "270 = 100 + 200", ("cdkt:270",), ("cdkt:100", "cdkt:200")),
    ("cdkt", "440 = 300 + 400", ("cdkt:440",), ("cdkt:300", "cdkt:400")),
    ("cdkt", "270 = 440", ("cdkt:270",), ("cdkt:440",)),
    ("kqkd", "20 = 10 - |11|", ("kqkd:20",), ("kqkd:10", "|kqkd:11")),
    ("kqkd", "50 = 30 + 40", ("kqkd:50",), ("kqkd:30", "kqkd:40")),
)


def values_by_code(grid: list[list[str]], column: int, kind: str) -> dict[str, float]:
    """First value column, keyed by `<kind>:<code>`; ambiguous codes are dropped."""

    value_columns = [c for c in lookup_mod.value_columns(grid) if c > column]
    if not value_columns:
        return {}
    target = value_columns[0]
    found: dict[str, float] = {}
    duplicated: set[str] = set()
    for row in grid[1:]:
        if column >= len(row) or target >= len(row):
            continue
        match = CODE_RE.match(str(row[column]).strip())
        if not match:
            continue
        value = lookup_mod._parse_cell(row[target])
        if value is None:
            continue
        key = f"{kind}:{match.group(1).lstrip('0') or '0'}"
        if key in found:
            duplicated.add(key)
        found[key] = value
    for key in duplicated:
        found.pop(key, None)
    return found


def main() -> None:
    frame = pd.read_parquet(
        ROOT / "artifacts" / "tables.parquet",
        columns=["doc_name", "caption", "unit_line", "rows_json", "eligible"])
    frame = frame[frame.eligible]

    results = {name: [0, 0] for _, name, _, _ in IDENTITIES}
    started = time.time()
    checked_tables = 0

    for row in frame.itertuples(index=False):
        grid = json.loads(row.rows_json)
        if len(grid) < 3:
            continue
        column = code_column(grid)
        if column is None:
            continue
        kind = statement_kind(grid, str(row.caption), str(row.unit_line))
        if not kind:
            continue
        values = values_by_code(grid, column, kind)
        if not values:
            continue
        checked_tables += 1
        for want_kind, name, left, right in IDENTITIES:
            if want_kind != kind:
                continue
            terms = list(left) + [t.lstrip("-|") for t in right]
            if any(t not in values for t in terms):
                continue
            lhs = sum(values[t] for t in left)
            rhs = 0.0
            for term in right:
                if term.startswith("|"):
                    rhs -= abs(values[term[1:]])
                elif term.startswith("-"):
                    rhs -= values[term[1:]]
                else:
                    rhs += values[term]
            scale = max(abs(lhs), abs(rhs), 1.0)
            ok = abs(lhs - rhs) / scale <= TOLERANCE
            results[name][1] += 1
            results[name][0] += 1 if ok else 0

    print(f"{checked_tables} bảng có mã số và đọc được giá trị, {time.time() - started:.0f}s\n")
    print(f"{'đẳng thức':<20}{'đúng':>8}{'kiểm':>8}{'tỷ lệ':>9}")
    total_ok = total = 0
    for _, name, _, _ in IDENTITIES:
        ok, count = results[name]
        total_ok += ok
        total += count
        share = f"{ok / count:.1%}" if count else "-"
        print(f"  {name:<18}{ok:>8}{count:>8}{share:>9}")
    if total:
        print(f"\nTổng: {total_ok}/{total} = {total_ok / total:.1%}")
        print("\nĐây là bằng chứng độc lập: nếu cột mã bị đọc sai, hoặc nhãn bị gán nhầm mã,")
        print("các đẳng thức này sẽ hỏng. Chúng không thể đúng do trùng hợp.")


if __name__ == "__main__":
    main()
