"""Row accuracy of the real `find()` path against the organisers' own coordinates.

The earlier version of this measurement called `match_row` directly, which left the
existing total-row fallback out of the baseline and so overstated what the caption
labelling adds. `find()` is the function the submission actually calls: metric
variants, the total-row rule, the column choice, all of it.

Gold rows come from the generator's `pandas_query`, which names the cell outright.
The gold table is used, so only the reading step is measured — retrieval is a
separate question with a separate number.

Usage:
  VIFIN_BLANK_CAPTION=1 VIFIN_MIN_LABEL_SCORE=0.55 \
    PYTHONPATH=src python scripts/_probe_find_rows.py --limit 1500
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

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

ILOC_NAME = re.compile(r"\.iloc\[\s*(-?\d+)\s*\]\s*\[\s*(['\"])(.*?)\2\s*\]", re.S)
ILOC_RC = re.compile(r"\.iloc\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def gold_row_of(query: str, grid) -> int | None:
    match = ILOC_NAME.search(query) or ILOC_RC.search(query)
    if match is None:
        return None
    index = int(match.group(1))
    row = len(grid) + index if index < 0 else index + 1
    return row if 0 <= row < len(grid) else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")

    tally: collections.Counter[str] = collections.Counter()
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
        key = TableKey(doc, int(table_id))
        grid = store.rows(key)
        if not grid or len(grid) < 2:
            continue
        gold_row = gold_row_of(record.get("pandas_query") or "", grid)
        if gold_row is None:
            continue
        seen += 1
        blank = not str(grid[gold_row][0]).strip()
        caption = str(getattr(store.meta(key), "caption", ""))[:60]
        parsed = parse_question(record.get("id", 0), record["question"], roster)
        found = lookup_mod.find(grid, parsed, caption)
        if found is None:
            tally["im lặng"] += 1
            tally["im lặng (gold ở dòng rỗng)"] += 1 if blank else 0
            continue
        tally["cam kết"] += 1
        if found.row == gold_row:
            tally["ĐÚNG dòng"] += 1
            if blank:
                tally["đúng ở dòng rỗng"] += 1
        else:
            tally["sai dòng"] += 1

    commit = max(tally["cam kết"], 1)
    print(f"{seen} bản ghi gold | caption={lookup_mod.BLANK_ROW_CAPTION} "
          f"ngưỡng={lookup_mod.MIN_LABEL_SCORE}")
    print(f"  cam kết      {tally['cam kết']:5d}  {tally['cam kết'] / max(seen, 1):6.1%}")
    print(f"  ĐÚNG dòng    {tally['ĐÚNG dòng']:5d}  {tally['ĐÚNG dòng'] / max(seen, 1):6.1%} "
          f"toàn bộ | {tally['ĐÚNG dòng'] / commit:.1%} trên số cam kết")
    print(f"  đúng ở dòng rỗng {tally['đúng ở dòng rỗng']:5d}")


if __name__ == "__main__":
    main()
