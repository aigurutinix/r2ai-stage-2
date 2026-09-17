"""Replace the note headings with the numbered heading, not the page furniture.

The block renderer shows each tied note by its heading so a model can pick one, and
the headings it currently shows are mostly useless:

  [TM13] nối cdkt/131  ngày 22/12/2014 của Bộ Tài chính) | 6. Phải thu ngắn hạn …
  [TM22] nối cdkt/313  (Ban hành theo Thông tư số 200/2014/TT-BTC ngày 22/12/2014 …
  [TM15] nối cdkt/136  (iii) Các khoản cho vay này không có tài sản đảm bảo, hưởng …

The first buries the real heading behind a circular reference, the second is nothing
but boilerplate, and the third is a narrative sentence from the previous note. That
happened because the heading was taken as the last three non-empty lines before the
`[table_N]` anchor, whatever they said.

What identifies a note is its numbered heading — "6. Phải thu ngắn hạn của khách
hàng", "24. Giá vốn hàng bán", "19. Vốn cổ phần". So the extractor looks back further
and picks the LAST numbered heading that is not form boilerplate, falling back to the
last short non-boilerplate line only when there is none.

Patches `notes.jsonl` in place, reading each report's text once rather than rerunning
the tie pass over 148,000 csv files.

Usage:  python scripts/fresh/fix_headings.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PAGE_RE = re.compile(r"===== PAGE (\d+) =====")
ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
# "6.", "14.", "5.3.", "II." — the numbering a note heading carries.
NUMBERED_RE = re.compile(r"^\s*(?:\(?[a-z]\)|[IVX]+\.|\d{1,2}(?:\.\d{1,2})*\.?)\s+"
                         r"(\S.{2,90})$")
BOILERPLATE = (
    re.compile(r"ban hành theo|thông tư số|bộ tài chính|mẫu\s*b\s*\d|mẫu số b", re.I),
    re.compile(r"thuyết minh (?:báo cáo )?tài chính|tiếp theo|\(tiếp\)", re.I),
    re.compile(r"^\s*(?:công ty|tổng công ty|ngân hàng|tập đoàn)\b.{0,60}$", re.I),
    re.compile(r"đơn vị|^\s*trang\s*\d|^\s*\d+\s*$", re.I),
    re.compile(r"bộ phận hợp thành|cần được đọc đồng thời|đính kèm", re.I),
    re.compile(r"^\s*(?:cho năm|năm tài chính|vào ngày|tại ngày)\b", re.I),
)


def is_boilerplate(line: str) -> bool:
    return any(pattern.search(line) for pattern in BOILERPLATE)


def headings_for(text: str, *, look_back: int = 14) -> dict[int, str]:
    """Per table id, the numbered heading closest above its anchor."""

    marks = list(PAGE_RE.finditer(text))
    pages = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        pages.append(text[mark.end():end])
    if not pages:
        pages = [text]

    out: dict[int, str] = {}
    for body in pages:
        lines = body.splitlines()
        # Where each anchor sits, by line number.
        anchor_line: dict[int, int] = {}
        for number, line in enumerate(lines):
            for match in ANCHOR_RE.finditer(line):
                anchor_line[int(match.group(1))] = number
        for table_id, position in anchor_line.items():
            window = [ANCHOR_RE.sub("", line).strip()
                      for line in lines[max(0, position - look_back):position]]
            window = [line for line in window if line and not is_boilerplate(line)]
            chosen = ""
            for line in reversed(window):
                match = NUMBERED_RE.match(line)
                if match:
                    chosen = match.group(1).strip()
                    break
            if not chosen:
                # No numbered heading: the shortest recent line is more likely a
                # title than a sentence from the note above.
                short = [line for line in window[-4:] if len(line) <= 90]
                chosen = short[-1] if short else (window[-1][:90] if window else "")
            out[table_id] = " ".join(chosen.split())[:110]
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", default="artifacts/fresh/notes.jsonl")
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    path = ROOT / args.notes
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_doc[record["doc"]].append(record)
    print(f"{len(records)} thuyet minh trong {len(by_doc)} bao cao", flush=True)

    changed = 0
    empty = 0
    started = time.time()
    for index, (doc, group) in enumerate(sorted(by_doc.items()), start=1):
        ticker = group[0]["ticker"]
        year = group[0]["year"]
        text_path = (ROOT / "data" / "official_corpus" / ticker / year / doc /
                     f"{doc}_extracted.txt")
        if not text_path.exists():
            continue
        table = headings_for(text_path.read_text(encoding="utf-8", errors="replace"))
        for record in group:
            heading = table.get(record["table_id"], "")
            if heading and heading != record.get("heading"):
                record["heading"] = heading
                changed += 1
            elif not heading:
                empty += 1
        if index % 300 == 0:
            print(f"  {index}/{len(by_doc)}  {time.time() - started:.0f}s", flush=True)

    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in records), encoding="utf-8")
    print(f"doi tieu de: {changed}, khong tim duoc: {empty}")
    print("\nvi du sau khi sua:")
    for record in records[:args.show]:
        print(f"  TM{record['table_id']:<4d} noi {','.join(record['paired'][:2]):<16s} "
              f"{record['heading'][:80]}")


if __name__ == "__main__":
    main()
