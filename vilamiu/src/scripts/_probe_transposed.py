"""Is the question's metric the name of a COLUMN rather than of a row?

48% of gold records have a gold row whose label shares no words with the question's
metric. Reading them showed why, and it is not bad gold:

  "Giá gốc"             -> gold row "Công ty Cổ phần Chứng khoán MB"
  "Tổng tài sản"        -> gold row "Hoạt động mua nợ"
  "Tổng vốn chủ sở hữu" -> gold row "Số dư cuối năm nay"
  "tiền gửi tại NHNN"   -> gold row "Bằng VND"

These tables are matrices. The rows are companies, segments, equity movements or
currencies, and the metric the question names is a COLUMN header: an investment note
lists holdings down the side and "Giá gốc"/"Dự phòng" across the top, a segment report
lists segments down the side and "Tổng tài sản" across the top.

Every mechanism in this project matches the metric against row labels and then picks a
column by period. For a matrix table that is the wrong way round, and no tuning of the
row matcher can fix it — which is why widening its reach, lowering its threshold and
relabelling its blank rows all came back flat.

This measures how many gold records are of that shape.

Usage:
  PYTHONPATH=src python scripts/_probe_transposed.py --limit 1500
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

ILOC_NAME = re.compile(r"\.iloc\[\s*(-?\d+)\s*\]\s*\[\s*(['\"])(.*?)\2\s*\]", re.S)
ILOC_RC = re.compile(r"\.iloc\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
# A header that only names a period is not a metric name.
PERIOD_RE = re.compile(
    r"^\s*((19|20)\d{2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}|số (cuối|đầu) (năm|kỳ)|"
    r"năm (nay|trước)|(cuối|đầu) (năm|kỳ)|mã số|thuyết minh)", re.I)


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def overlap(metric: str, label: str) -> float:
    wanted = {t for t in fold(metric).split() if len(t) > 2}
    have = {t for t in fold(label).split() if len(t) > 2}
    if not wanted or not have:
        return 0.0
    shared = len(wanted & have)
    if not shared:
        return 0.0
    coverage = shared / len(wanted)
    focus = shared / len(have)
    return 2 * coverage * focus / (coverage + focus)


def gold_cell(query: str, grid):
    """(row, column) in grid coordinates from the generator's own program."""

    match = ILOC_NAME.search(query)
    if match:
        index = int(match.group(1))
        row = len(grid) + index if index < 0 else index + 1
        name = match.group(3)
        header = [str(cell) for cell in grid[0]]
        squashed = [re.sub(r"\s+", "", cell) for cell in header]
        target = re.sub(r"\s+", "", name)
        column = (header.index(name) if name in header
                  else (squashed.index(target) if target in squashed else None))
        return (row, column) if 0 <= row < len(grid) else None
    match = ILOC_RC.search(query)
    if match:
        index = int(match.group(1))
        row = len(grid) + index if index < 0 else index + 1
        column = int(match.group(2))
        if column < 0:
            column = len(grid[0]) + column
        return (row, column) if 0 <= row < len(grid) else None
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    tally: collections.Counter[str] = collections.Counter()
    examples = []
    seen = 0

    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or seen >= args.limit:
            continue
        record = json.loads(line)
        refs = record.get("relevant_tables") or []
        if len(refs) != 1:
            continue
        doc, _, table_id = refs[0].rpartition("|table_")
        if not table_id.isdigit():
            continue
        grid = store.rows(TableKey(doc, int(table_id)))
        if not grid or len(grid) < 2:
            continue
        located = gold_cell(record.get("pandas_query") or "", grid)
        if located is None:
            continue
        row, column = located
        seen += 1
        metric = lookup_mod.extract_metric(record["question"])

        best_row = max((overlap(metric, str(line[0])) for line in grid[1:] if line),
                       default=0.0)
        headers = [(index, str(cell)) for index, cell in enumerate(grid[0])
                   if str(cell).strip() and not PERIOD_RE.match(str(cell).strip())]
        best_header, best_header_score = None, 0.0
        for index, text in headers:
            value = overlap(metric, text)
            if value > best_header_score:
                best_header, best_header_score = index, value

        if best_row >= 0.6:
            tally["chỉ tiêu là TÊN DÒNG"] += 1
        elif best_header_score >= 0.6:
            tally["CHỈ TIÊU LÀ TÊN CỘT"] += 1
            if column is not None and best_header == column:
                tally["  và trùng cột gold"] += 1
            if len(examples) < args.show:
                examples.append((record.get("id"), metric[:38],
                                 str(grid[row][0])[:32],
                                 str(grid[0][best_header])[:28],
                                 column, best_header))
        elif best_row >= 0.3 or best_header_score >= 0.3:
            tally["khớp một phần, chưa rõ trục"] += 1
        else:
            tally["không khớp trục nào"] += 1

    print(f"{seen} bản ghi gold\n")
    for name, count in tally.most_common():
        print(f"  {name:32s} {count:5d}  {count / max(seen, 1):6.1%}")
    for qid, metric, row_label, header, gold_col, picked_col in examples:
        print(f"\n  id={qid} hỏi {metric!r}")
        print(f"     dòng gold: {row_label!r}")
        print(f"     cột khớp chỉ tiêu: {header!r} (c{picked_col}), "
              f"cột gold = c{gold_col}")


if __name__ == "__main__":
    main()
