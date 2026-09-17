"""Recover the note heading each table sits under, because `caption` mostly is not it.

Measured on the gold tables: only 24.9% of `caption` values are a note heading. The
rest are form codes ("B 09 - DN/HN", 14.3%), company names (12.0%), date lines (4.0%),
addresses, page footers — and a large "other" bucket that mixes real descriptions with
more furniture. Across the whole index only 22.5% are headings.

That field is what the retrieval index is built on and what table selection scores, so
both have been working off page furniture. It also sank the two-stage experiment: the
model's predicted headings ("VỐN CHỦ SỞ HỮU", "PHÁT HÀNH GIẤY TỜ CÓ GIÁ") were correct
and had nothing to match against.

The heading is in the text, a few lines above the table's anchor:

    5.9 Chi phí xây dựng cơ bản dở dang
    [table_12](…/table_12.csv)

So this walks upward from the anchor and takes the first line that is a heading rather
than furniture. `start_line` in the store is the 1-based anchor line in
`data/official_corpus/**/*_extracted.txt`, which is where the anchors live.

Two shapes count as a heading, in this order of preference:
  numbered — "9. CHO VAY KHÁCH HÀNG", "24.3 Chi tiết cổ phiếu của Ngân hàng"
  bare     — a descriptive sentence fragment naming the table, e.g. "Phân tích số dư
             tiền gửi của khách hàng theo đối tượng khách hàng"

Usage:
  PYTHONPATH=src python scripts/build_note_titles.py --before 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.store import TableStore  # noqa: E402

CORPUS = ROOT / "data" / "official_corpus"

ANCHOR_RE = re.compile(r"\[table_(\d+)\]\(")
PAGE_RE = re.compile(r"^=+\s*PAGE\s+\d+\s*=+$", re.I)
NUMBERED_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2})*)\.?\s+(\S.*)$")

# Everything below is furniture: it appears above most tables and names no table.
FURNITURE = (
    re.compile(r"^\s*(Đ|E)ịa\s*chỉ", re.I),
    re.compile(r"^\s*(Lô|Km|Tầng|Phòng|Số)\s+\d", re.I),
    re.compile(r"^\s*(CÔNG TY|TỔNG CÔNG TY|NGÂN HÀNG|TẬP ĐOÀN|CTCP|QUỸ)\b", re.I),
    re.compile(r"mẫu\s*(số)?\s*b\s*\d|ban hành theo|thông tư số|^\s*B\s*\d{2}\s*-\s*DN", re.I),
    re.compile(r"^\s*(tại ngày|cho năm|từ ngày|cho kỳ|ngày \d{1,2}\s*tháng)", re.I),
    re.compile(r"thuyết minh (này|từ trang)|bộ phận hợp thành|đọc đồng thời với", re.I),
    re.compile(r"^\s*(THUYẾT MINH|BÁO CÁO|BẢNG CÂN ĐỐI)\s+(BÁO CÁO|TÀI CHÍNH|KẾT QUẢ|LƯU CHUYỂN|KẾ TOÁN)", re.I),
    re.compile(r"^\s*(Đơn vị( tính)?|Unit)\s*[::]", re.I),
    re.compile(r"^\s*\(?(tiếp theo|đã kiểm toán|soát xét)\)?\s*$", re.I),
    re.compile(r"^\s*[\d.,()%\-\s]+$"),          # a stray row of figures
    re.compile(r"^\s*\S{1,3}\s*$"),               # a page number or single glyph
)

# OCR garbage: a line with almost no Vietnamese letters is not a heading.
LETTERS_RE = re.compile(r"[a-zA-ZÀ-ỹ]")


def is_furniture(line: str) -> bool:
    return any(pattern.search(line) for pattern in FURNITURE)


def readable(line: str) -> bool:
    letters = len(LETTERS_RE.findall(line))
    return letters >= 8 and letters / max(len(line), 1) > 0.45


def document_path(doc_name: str):
    parts = doc_name.split("_")
    year = next((p for p in parts if p.isdigit() and len(p) == 4), None)
    if not parts or year is None:
        return None
    path = CORPUS / parts[0] / year / doc_name / f"{doc_name}_extracted.txt"
    return path if path.exists() else None


def find_heading(lines: list[str], anchor: int, table_id: int, before: int):
    """(heading, kind) for the table anchored at 1-based line `anchor`."""

    numbered = bare = None
    start = max(0, anchor - 1 - before)
    for index in range(anchor - 2, start - 1, -1):
        if index < 0 or index >= len(lines):
            continue
        text = lines[index].strip()
        if not text:
            continue
        found = ANCHOR_RE.search(text)
        if found and int(found.group(1)) != table_id:
            # Anything above the previous table's anchor describes that table.
            break
        if PAGE_RE.match(text):
            # A page break is not a boundary for the note, only for the page.
            continue
        if is_furniture(text) or not readable(text):
            continue
        match = NUMBERED_RE.match(text)
        if match and len(match.group(2).split()) >= 2:
            numbered = f"{match.group(1)} {match.group(2)}".strip()
            break
        if bare is None and len(text.split()) >= 3:
            bare = text
    if numbered:
        return numbered[:150], "numbered"
    if bare:
        return bare[:150], "bare"
    return "", "none"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=int, default=40)
    parser.add_argument("--out", default="artifacts/table_notes.jsonl")
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    frame = store.frame[store.frame.eligible.astype(bool)]
    cache: dict[str, list[str]] = {}
    kinds = {"numbered": 0, "bare": 0, "none": 0}
    written = 0
    examples = []

    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for record in frame.itertuples():
            doc = str(record.doc_name)
            if doc not in cache:
                path = document_path(doc)
                cache[doc] = (path.read_text(encoding="utf-8", errors="replace")
                              .splitlines() if path else [])
            lines = cache[doc]
            if not lines:
                kinds["none"] += 1
                continue
            heading, kind = find_heading(lines, int(record.start_line),
                                         int(record.table_id), args.before)
            kinds[kind] += 1
            if not heading:
                continue
            handle.write(json.dumps(
                {"doc": doc, "table_id": int(record.table_id),
                 "note": heading, "kind": kind}, ensure_ascii=False) + "\n")
            written += 1
            if len(examples) < args.show and kind == "numbered":
                examples.append((doc[:34], int(record.table_id), heading[:70]))

    total = max(sum(kinds.values()), 1)
    print(f"{written} bảng có tiêu đề -> artifacts/{Path(args.out).name}")
    for name in ("numbered", "bare", "none"):
        print(f"  {name:10s} {kinds[name]:7d}  {kinds[name] / total:6.1%}")
    for doc, table_id, heading in examples:
        print(f"\n  {doc} t{table_id}")
        print(f"     {heading!r}")


if __name__ == "__main__":
    main()
