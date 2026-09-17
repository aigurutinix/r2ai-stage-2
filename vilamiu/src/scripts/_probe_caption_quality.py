"""Is the `caption` field a note heading, or is it page furniture?

The two-stage idea — a first pass names the note, a second pass finds the cell inside
it — measured worse than the baseline: 27.3% against 41.7%. Reading the pairs shows
why, and it is not the idea:

  model said  "VỐN CHỦ SỞ HỮU"                  gold caption "Eịa chỉ: 230 Đại lộ Bình Dương…"
  model said  "TÀI SẢN CỐ ĐỊNH HỮU HÌNH"        gold caption "CÔNG TY CỔ PHẦN … B 09 - DN/HN | Địa chỉ: Km 11+5…"
  model said  "PHÁT HÀNH GIẤY TỜ CÓ GIÁ"        gold caption "24.3 Chi tiết cổ phiếu của Ngân hàng"

The model's headings are clean and correct. The captions they were scored against are
company addresses, form numbers and date lines. So the field this project has used as
"the note this table belongs to" — in the retrieval index, in the table-selection
score, in every measurement of both — is frequently not that at all.

A real note heading has a recognisable shape in these reports: a number, then a noun
phrase, e.g. "9. CHO VAY KHÁCH HÀNG", "24.3 Chi tiết cổ phiếu của Ngân hàng". Page
furniture does not: it starts with an address, a company name, a form code, or a date.

This counts which is which, so the size of the defect is known before anything is
rebuilt on top of it.

Usage:
  PYTHONPATH=src python scripts/_probe_caption_quality.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.store import TableKey, TableStore  # noqa: E402

# "9. CHO VAY KHÁCH HÀNG", "24.3 Chi tiết cổ phiếu", "5.1. Tiền và tương đương tiền".
NOTE_HEADING_RE = re.compile(r"^\s*\d{1,2}(\.\d{1,2})*\.?\s+\S")
# What page furniture opens with.
ADDRESS_RE = re.compile(r"^\s*(Đ|E)ịa chỉ|^\s*Số \d+[,/]|^\s*(Lô|Km|Tầng|Phòng)\s", re.I)
COMPANY_RE = re.compile(r"^\s*(CÔNG TY|TỔNG CÔNG TY|NGÂN HÀNG|TẬP ĐOÀN|CTCP)", re.I)
FORM_RE = re.compile(r"(mẫu\s*(số)?\s*b\s*\d|ban hành theo|thông tư số|"
                     r"^\s*B\s*\d{2}\s*-\s*DN)", re.I)
DATE_RE = re.compile(r"^\s*(tại ngày|cho năm|từ ngày|ngày \d{1,2}\s*tháng)", re.I)
FOOTER_RE = re.compile(r"thuyết minh (này|từ trang)|bộ phận hợp thành|"
                       r"đọc đồng thời với", re.I)


def classify(caption: str) -> str:
    text = str(caption).strip()
    if not text:
        return "rỗng"
    if NOTE_HEADING_RE.match(text):
        return "TIÊU ĐỀ THUYẾT MINH"
    if FOOTER_RE.search(text):
        return "chân trang"
    if ADDRESS_RE.match(text):
        return "địa chỉ"
    if COMPANY_RE.match(text):
        return "tên công ty"
    if FORM_RE.search(text):
        return "số mẫu biểu"
    if DATE_RE.match(text):
        return "dòng ngày tháng"
    return "khác"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--show", type=int, default=4)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    prose: dict[tuple[str, int], str] = {}
    prose_path = ROOT / "artifacts" / "table_prose.jsonl"
    if prose_path.exists():
        for line in prose_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prose[(row["doc"], int(row["table_id"]))] = row["prose"]

    # All eligible tables, so the index-wide picture is visible, and separately the
    # gold tables, because those are the ones selection has to get right.
    everywhere: collections.Counter[str] = collections.Counter()
    frame = store.frame[store.frame.eligible.astype(bool)]
    for record in frame.itertuples():
        everywhere[classify(record.caption)] += 1

    on_gold: collections.Counter[str] = collections.Counter()
    prose_helps = 0
    examples: dict[str, list] = collections.defaultdict(list)
    seen = 0
    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        refs = record.get("relevant_tables") or []
        if len(refs) != 1:
            continue
        doc, _, table_id = refs[0].rpartition("|table_")
        if not table_id.isdigit():
            continue
        key = TableKey(doc, int(table_id))
        caption = str(getattr(store.meta(key), "caption", ""))
        kind = classify(caption)
        on_gold[kind] += 1
        seen += 1
        if kind != "TIÊU ĐỀ THUYẾT MINH":
            text = prose.get((doc, int(table_id)), "")
            # Does the prose window contain a heading the caption missed?
            if any(NOTE_HEADING_RE.match(part.strip())
                   for part in re.split(r"(?<=[.:])\s+", text)):
                prose_helps += 1
            if len(examples[kind]) < args.show:
                examples[kind].append((record.get("id"), caption[:70], text[:70]))

    total = max(sum(everywhere.values()), 1)
    print(f"TOÀN BỘ CHỈ MỤC — {total} bảng eligible")
    for name, count in everywhere.most_common():
        print(f"  {name:24s} {count:7d}  {count / total:6.1%}")

    print(f"\nBẢNG GOLD — {seen} bảng")
    for name, count in on_gold.most_common():
        print(f"  {name:24s} {count:7d}  {count / max(seen, 1):6.1%}")
    bad = seen - on_gold.get("TIÊU ĐỀ THUYẾT MINH", 0)
    print(f"\n  caption KHÔNG phải tiêu đề thuyết minh: {bad}/{seen} = "
          f"{bad / max(seen, 1):.1%}")
    print(f"  trong số đó, văn xuôi có chứa một tiêu đề: {prose_helps}/{bad} = "
          f"{prose_helps / max(bad, 1):.1%}")
    for name, rows in examples.items():
        for qid, caption, text in rows[:2]:
            print(f"\n  [{name}] id={qid}")
            print(f"     caption: {caption!r}")
            print(f"     prose  : {text!r}")


if __name__ == "__main__":
    main()
