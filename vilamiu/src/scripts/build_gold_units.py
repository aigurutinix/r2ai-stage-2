"""Write a gold set whose answers are in the unit each question asks for.

`easy_full.jsonl` stores the raw cell. Where the question asks for "tỷ đồng" and
the table prints VND, the stored answer is a thousand million times the right one,
which happens on 32.4% of records. Scoring against it marks a correct conversion
wrong and a raw echo right — and the leaderboard scores the opposite way, so every
local measurement built on the raw file understates the pipeline by 8 points.

This was found once before and worked around by comparing cell positions instead
of values. That kept the aggregate honest but left the file wrong, and the next
measurement to compare values walked into it again. Writing the corrected file
ends that.

A record is rescaled only when its answer is literally a cell of its own gold
table; a derived answer is already in whatever unit the generator chose.

Usage:  PYTHONPATH=src python scripts/build_gold_units.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _rescore_unit_fixed import asked_unit, parse_number, table_scale  # noqa: E402

from vifin.store import TableKey, TableStore  # noqa: E402

SOURCE = ROOT / "artifacts" / "easy_full.jsonl"
OUT = ROOT / "artifacts" / "easy_full_units.jsonl"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    written = rescaled = untouched = 0
    lines = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        answer = parse_number(record.get("answer"))
        refs = record.get("relevant_tables") or []
        wanted = asked_unit(record.get("question", ""))
        new_answer = None
        if answer is not None and refs and wanted is not None:
            doc, tid = refs[0].rsplit("|table_", 1)
            grid = store.rows(TableKey(doc, int(tid)))
            if grid:
                is_cell = any(
                    parse_number(cell) == answer
                    for row in grid[1:] for cell in row
                    if parse_number(cell) is not None
                )
                if is_cell:
                    meta = store.meta(TableKey(doc, int(tid)))
                    scale = table_scale(grid, getattr(meta, "caption", ""))
                    if abs(scale - wanted) > 1:
                        new_answer = round(answer * scale / wanted, 2)
        if new_answer is None:
            untouched += 1
        else:
            rescaled += 1
            record["answer_raw_cell"] = record["answer"]
            record["answer"] = new_answer
        lines.append(json.dumps(record, ensure_ascii=False))
        written += 1

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{written} records -> {OUT}")
    print(f"  rescaled to the asked unit : {rescaled} ({100 * rescaled / max(written,1):.1f}%)")
    print(f"  left as stored             : {untouched}")
    print("  the original value is kept in `answer_raw_cell` on rescaled records")


if __name__ == "__main__":
    main()
