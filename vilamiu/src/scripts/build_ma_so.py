"""Derive the statutory line-item code dictionary from the corpus itself.

Vietnamese statutory statements carry a "Mã số" column: 100 is Tài sản ngắn hạn,
270 is Tổng cộng tài sản, and so on under Circular 200. The code is a legal
identifier, not a phrasing, so matching on it is *exact* where matching on the
row label is fuzzy — and label matching is what caps our answering path at 42%.

Nothing here reads the organisers' package. The mapping is measured from the
146,246 extracted tables: find the column that is dominated by 2-3 digit
integers, pair each code with the row label beside it, and count. What comes out
is ours, it is wider than any hand-written catalogue, and it can be checked
against the accounting identities we already verify (TS = TSNH + TSDH).

Statement kind is inferred from the table's own caption rather than from the code
range, because KQKD and LCTT both number from 01 and would otherwise collide.

Usage:  PYTHONPATH=src python scripts/build_ma_so.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

CODE_RE = re.compile(r"^\s*(\d{2,3})\s*$")
# A label needs letters; a bare number or a lone bullet is not a line item.
LETTER_RE = re.compile(r"[A-Za-zÀ-ỹ]")

MIN_CODED_ROWS = 4
CODE_SHARE = 0.7
# Only the first few columns can be the code column; later ones are amounts.
MAX_CODE_COLUMN = 5


def plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).strip().casefold()


# Anchor line items that only ever appear in one statement. Classifying by the
# caption alone recognised 3,056 of 11,713 coded tables and returned 2,692 cash
# flow statements against 331 balance sheets — impossible, since every report has
# both. The heading usually sits in the page text above the table, not inside it,
# so the statement has to be identified from the rows themselves.
KIND_ANCHORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lctt", ("luu chuyen tien thuan", "tien va tuong duong tien dau",
              "tien chi mua sam", "tien thu tu di vay")),
    ("cdkt", ("tong cong tai san", "tong cong nguon von", "tai san ngan han",
              "no phai tra", "von chu so huu")),
    ("kqkd", ("doanh thu thuan", "gia von hang ban", "loi nhuan gop",
              "chi phi ban hang", "loi nhuan sau thue")),
)


def statement_kind(grid: list[list[str]], caption: str, unit_line: str) -> str:
    """Which statement the table belongs to, from its own rows then its caption."""

    text = plain(" ".join(
        str(cell) for row in grid for cell in row[:3] if LETTER_RE.search(str(cell))
    ))
    scores = {
        kind: sum(1 for anchor in anchors if anchor in text)
        for kind, anchors in KIND_ANCHORS
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= 2:
        return best

    heading = plain(f"{caption} {unit_line}")
    if "luu chuyen tien" in heading:
        return "lctt"
    if "ket qua hoat dong kinh doanh" in heading or "ket qua kinh doanh" in heading:
        return "kqkd"
    if "can doi ke toan" in heading or "tinh hinh tai chinh" in heading:
        return "cdkt"
    return ""


def code_column(grid: list[list[str]]) -> int | None:
    width = max((len(row) for row in grid), default=0)
    for column in range(min(width, MAX_CODE_COLUMN)):
        values = [
            str(row[column]).strip()
            for row in grid[1:]
            if column < len(row) and str(row[column]).strip()
        ]
        if len(values) < MIN_CODED_ROWS:
            continue
        coded = sum(1 for v in values if CODE_RE.match(v))
        if coded >= max(MIN_CODED_ROWS, int(CODE_SHARE * len(values))):
            return column
    return None


def label_for(row: list[str], code_at: int) -> str:
    """The nearest cell left of the code that reads like a line item."""

    for column in range(code_at - 1, -1, -1):
        if column < len(row):
            text = str(row[column]).strip()
            if text and LETTER_RE.search(text):
                return text
    return ""


def main() -> None:
    frame = pd.read_parquet(
        ROOT / "artifacts" / "tables.parquet",
        columns=["doc_name", "ticker", "year", "table_id", "caption", "unit_line",
                 "rows_json", "eligible"])
    frame = frame[frame.eligible]
    print(f"{len(frame)} bảng eligible")

    seen: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    docs: dict[str, set] = collections.defaultdict(set)
    kinds: collections.Counter = collections.Counter()
    tables_with_code = 0
    unknown_kind = 0
    started = time.time()

    for position, row in enumerate(frame.itertuples(index=False)):
        grid = json.loads(row.rows_json)
        if len(grid) < 3:
            continue
        column = code_column(grid)
        if column is None:
            continue
        kind = statement_kind(grid, str(row.caption), str(row.unit_line))
        if not kind:
            unknown_kind += 1
            continue
        tables_with_code += 1
        kinds[kind] += 1
        for line in grid[1:]:
            if column >= len(line):
                continue
            match = CODE_RE.match(str(line[column]).strip())
            if not match:
                continue
            label = label_for(line, column)
            if not label:
                continue
            key = f"{kind}:{match.group(1).lstrip('0') or '0'}"
            seen[key][label] += 1
            docs[key].add(row.doc_name)
        if position % 20000 == 0 and position:
            print(f"  {position} bảng, {len(seen)} mã, {time.time() - started:.0f}s")

    print(f"\n{tables_with_code} bảng có cột Mã số và nhận dạng được loại báo cáo")
    print(f"{unknown_kind} bảng có mã nhưng không rõ loại báo cáo (bỏ qua)")
    print("theo loại:", dict(kinds))
    print(f"{len(seen)} mã phân biệt, {time.time() - started:.0f}s")

    out = ROOT / "artifacts" / "ma_so.json"
    payload = {}
    for key, counter in seen.items():
        variants = counter.most_common()
        payload[key] = {
            "canonical": variants[0][0],
            "documents": len(docs[key]),
            "occurrences": sum(counter.values()),
            "variants": [{"label": label, "count": count} for label, count in variants[:25]],
        }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"đã ghi {out}")

    strongest = sorted(payload.items(), key=lambda kv: -kv[1]["documents"])[:20]
    print(f"\n{'mã':<12}{'tài liệu':>9}  nhãn phổ biến nhất  (số biến thể)")
    for key, entry in strongest:
        print(f"  {key:<10}{entry['documents']:>9}  {entry['canonical'][:52]!r} "
              f"({len(entry['variants'])})")


if __name__ == "__main__":
    main()
