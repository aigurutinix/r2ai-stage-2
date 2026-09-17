"""Read-only helper for auditing exact ViFinQA source-table cells.

Examples:
    python tools/audit_tables.py ACB 2015,2019,2022 "quy khen thuong"
    python tools/audit_tables.py ACB 2022 --table 28
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from road2ai_vifinqa.paths import INDEX_PATH
from road2ai_vifinqa.text import fold_text


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("years", help="Comma-separated report years")
    parser.add_argument("terms", nargs="*", help="All folded substrings must match the row")
    parser.add_argument("--scope", choices=("parent", "consolidated", "unknown"))
    parser.add_argument("--table", type=int)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    years = [int(value) for value in args.years.split(",")]
    conn = sqlite3.connect(f"file:{INDEX_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in years)
    params: list[object] = [args.ticker, *years]
    where = ["d.ticker=?", f"d.report_year IN ({placeholders})"]
    if args.scope:
        where.append("d.scope=?")
        params.append(args.scope)
    if args.table is not None:
        where.append("r.table_id=?")
        params.append(args.table)
    for term in args.terms:
        where.append("r.folded_text LIKE ?")
        params.append(f"%{fold_text(term)}%")
    query = f"""
        SELECT d.report_year, d.scope, r.doc_id, r.table_id, r.row_idx,
               r.cells_json, t.context
        FROM rows r
        JOIN documents d ON d.doc_id=r.doc_id
        JOIN tables t ON t.doc_id=r.doc_id AND t.table_id=r.table_id
        WHERE {' AND '.join(where)}
        ORDER BY d.report_year, d.scope, r.doc_id, r.table_id, r.row_idx
        LIMIT ?
    """
    params.append(args.limit)
    for row in conn.execute(query, params):
        cells = json.loads(row["cells_json"])
        print(
            f"{row['report_year']} {row['scope']} {row['doc_id']}|{row['table_id']} "
            f"r{row['row_idx']} ctx={row['context'][-160:]!r}\n  {cells!r}"
        )


if __name__ == "__main__":
    main()
