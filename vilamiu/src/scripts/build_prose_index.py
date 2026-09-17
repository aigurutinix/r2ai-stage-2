"""Cache the prose that introduces each table, for the retrieval index.

Table selection is the weakest measured step in the pipeline. On gold records the
right document is found 96.4% of the time, the right table is in the eight-table
shortlist 72.5% of the time, and the matcher then picks it 37.9% of the time — 27.5%
end to end. Every attempt to improve *reading* has come back flat, because reading is
not where the loss is.

Our index gives each table its caption, unit line, header row and first column. The
caption is whatever line sits directly above the table anchor, and in this corpus that
is frequently the page footer — "Các thuyết minh này là bộ phận hợp thành và cần được
đọc đồng thời với …" — while the note title that actually names the table sits a few
lines further up:

    1. THÔNG TIN KHÁI QUÁT (TIẾP THEO)
    Vốn điều lệ của Công ty
    Theo Giấy chứng nhận đăng ký kinh doanh thay đổi lần thứ 20 …
    [table_5](…/table_5.csv)

The organisers' own retriever indexes 2,500 characters of the page
(`encoding/table_text.py:69-78`), which is why their questions can avoid the table's
own wording — their generator rejects any question that copies a row label verbatim.

`start_line` in the store is the 1-based line of the table anchor in
`*_extracted.txt`, so the window is directly addressable.

Usage:
  PYTHONPATH=src python scripts/build_prose_index.py --before 30 --after 4
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

# Page furniture that appears above most tables and names nothing.
BOILERPLATE_RE = re.compile(
    r"thuyết minh này là bộ phận hợp thành|đọc đồng thời với|"
    r"^trang\s*\d+|^\d+\s*$|báo cáo tài chính (hợp nhất|riêng)?\s*$|"
    r"^\(tiếp theo\)$|^mẫu số|^đơn vị:", re.I)

# Another table's anchor: prose beyond it belongs to that table, not this one.
ANCHOR_RE = re.compile(r"\[table_(\d+)\]\(")


def document_path(doc_name: str) -> Path | None:
    """`AAA_financial_statements_2015_consolidated` -> its extracted .txt."""

    parts = doc_name.split("_")
    if len(parts) < 3:
        return None
    ticker = parts[0]
    year = next((p for p in parts if p.isdigit() and len(p) == 4), None)
    if year is None:
        return None
    path = CORPUS / ticker / year / doc_name / f"{doc_name}_extracted.txt"
    return path if path.exists() else None


def window(lines: list[str], anchor: int, before: int, after: int,
           table_id: int) -> str:
    """The prose introducing the table at 1-based line `anchor`.

    Walks upward from the anchor, dropping page furniture, and stops at the previous
    table's anchor so one note's prose is not attributed to the next.
    """

    start = max(0, anchor - 1 - before)
    collected: list[str] = []
    for index in range(anchor - 2, start - 1, -1):
        if index < 0 or index >= len(lines):
            continue
        text = lines[index].strip()
        if not text:
            continue
        found = ANCHOR_RE.search(text)
        if found and int(found.group(1)) != table_id:
            break
        if BOILERPLATE_RE.search(text):
            continue
        collected.append(text)
    collected.reverse()
    for index in range(anchor, min(anchor + after, len(lines))):
        text = lines[index].strip()
        if text and not ANCHOR_RE.search(text) and not BOILERPLATE_RE.search(text):
            collected.append(text)
    return " ".join(collected)[:1200]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=int, default=30)
    parser.add_argument("--after", type=int, default=4)
    parser.add_argument("--out", default="artifacts/table_prose.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    frame = store.frame[store.frame.eligible.astype(bool)]
    cache: dict[str, list[str]] = {}
    written = missing = empty = 0

    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for record in frame.itertuples():
            doc = str(record.doc_name)
            if doc not in cache:
                path = document_path(doc)
                cache[doc] = (path.read_text(encoding="utf-8", errors="replace")
                              .splitlines() if path else [])
            lines = cache[doc]
            if not lines:
                missing += 1
                continue
            text = window(lines, int(record.start_line), args.before, args.after,
                          int(record.table_id))
            if not text:
                empty += 1
                continue
            handle.write(json.dumps(
                {"doc": doc, "table_id": int(record.table_id), "prose": text},
                ensure_ascii=False) + "\n")
            written += 1

    print(f"{written} bảng có văn xuôi -> artifacts/{Path(args.out).name}")
    print(f"  không tìm thấy tài liệu: {missing}")
    print(f"  cửa sổ rỗng sau khi lọc: {empty}")


if __name__ == "__main__":
    main()
