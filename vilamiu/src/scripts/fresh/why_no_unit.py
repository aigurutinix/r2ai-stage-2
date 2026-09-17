"""Why do two thirds of note tables have no recognisable unit?

`scale_from_unit_text` needs a đồng token and a unit marker, and it is fed the table's
own header plus the lines on the anchor's page that START WITH one of "Đơn vị tính",
"Đơn vị tiền tệ", "Đơn vị", "ĐVT". Two thirds of note tables come back with no scale,
which blocks converting their cells to đồng and therefore blocks answering from them.

Rather than guess, this prints what is actually near those tables: every line on the
page mentioning a unit at all, and the table's own header. The candidate explanations
have different fixes:

  the prefix test    a line reading "(Đơn vị: VND)" starts with a bracket and fails
                     `startswith`, so a one-character fix recovers it
  declared elsewhere the notes section states the unit once and the tables inherit it,
                     so the fix is to look further — but carrying the unit across pages
                     was measured to make things worse, so it would need to be the
                     document's modal scale instead
  genuinely absent   the table states no unit anywhere, and the scale can only be
                     inferred — from the report's other tables, or from a total that
                     ties to a statement line

Usage:  python scripts/fresh/why_no_unit.py --n 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from build_index2 import page_units  # noqa: E402

PAGE_RE = re.compile(r"===== PAGE (\d+) =====")
ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
MENTIONS_UNIT = re.compile(r"đơn\s*vị|đvt|\bvnd\b|\bvnđ\b|triệu đồng|tỷ đồng|"
                           r"nghìn đồng", re.I)


def page_lines(text: str) -> dict[int, list[str]]:
    """Every line on the page each table anchor sits on."""

    marks = list(PAGE_RE.finditer(text))
    pages = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        pages.append(text[mark.end():end])
    if not pages:
        pages = [text]
    out: dict[int, list[str]] = {}
    for body in pages:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        for match in ANCHOR_RE.finditer(body):
            out[int(match.group(1))] = lines
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--docs", type=int, default=60)
    args = parser.parse_args()

    # Take note tables with no scale, grouped by document so each text is read once.
    wanted: dict[str, list[dict]] = {}
    for line in (ROOT / args.tables).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["scale"] is not None:
            continue
        wanted.setdefault(record["doc"], []).append(record)
        if len(wanted) > args.docs:
            break

    counters: Counter[str] = Counter()
    samples = []
    for doc, group in wanted.items():
        first = group[0]
        path = (ROOT / "data" / "official_corpus" / first["ticker"] /
                first["year"] / doc / f"{doc}_extracted.txt")
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines_by = page_lines(text)
        _table_page, strict = page_units(text)

        for record in group:
            table_id = record["table_id"]
            page = lines_by.get(table_id, [])
            mentions = [line for line in page if MENTIONS_UNIT.search(line)]
            strict_hit = strict.get(table_id, ())

            if not mentions:
                counters["trang KHONG he nhac don vi"] += 1
            elif strict_hit:
                # The strict filter found a line but `scale_from_unit_text` still
                # refused it — the wording is there and the parser rejects it.
                counters["co dong don vi nhung PARSER tu choi"] += 1
            else:
                counters["co nhac don vi nhung LOT bo loc startswith"] += 1
                if len(samples) < args.n:
                    samples.append((doc, table_id, record["heading"][:60],
                                    mentions[:2]))
    total = sum(counters.values())
    print(f"{total} bang thuyet minh khong ro don vi\n")
    for name, count in counters.most_common():
        print(f"  {name}: {count} ({100 * count / max(1, total):.0f}%)")
    print("\nvi du bi bo loc startswith loai:")
    for doc, table_id, heading, mentions in samples:
        print(f"  {doc[:34]} t{table_id}  {heading}")
        for line in mentions:
            print(f"     | {line[:96]}")


if __name__ == "__main__":
    main()
