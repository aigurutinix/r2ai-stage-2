"""Is the label column of the silent groups' statements simply not column 0?

In a silent group the best statement table anchors a median of one row label,
where a real balance sheet anchors many. That is not a threshold to lower — it
says the text being matched is not the line item. `extract_located` folds
`row[0]`, but `find_code_column` already concedes the code can sit anywhere, and
a layout that puts the code first and the name second would fail exactly this way.

This counts alias matches per column, for statements inside silent groups and,
as a control, inside groups the panel does serve.

Usage:  PYTHONPATH=src python scripts/_panel_label_column.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.corpus.metrics import (  # noqa: E402
    BY_STATEMENT, _fold, _matches_label, build_panel, detect_statement,
)
from vifin.store import TableKey, TableStore  # noqa: E402

MAX_COLUMN = 4


def matches_by_column(grid, statement) -> collections.Counter:
    catalogue = BY_STATEMENT[statement]
    hits: collections.Counter[int] = collections.Counter()
    for row in grid[1:]:
        for column in range(min(MAX_COLUMN, len(row))):
            folded = _fold(str(row[column]))
            if not folded:
                continue
            if any(_matches_label(folded, a) for m in catalogue for a in m.aliases):
                hits[column] += 1
    return hits


def survey(keys, groups, store, label: str, limit: int) -> None:
    best_column: collections.Counter[int] = collections.Counter()
    totals: collections.Counter[int] = collections.Counter()
    examined = 0
    for key in keys[:limit]:
        for row in groups[key]:
            grid = store.rows(TableKey(row.doc_name, int(row.table_id)))
            if len(grid) < 3:
                continue
            statement = detect_statement(grid)
            if statement is None:
                continue
            hits = matches_by_column(grid, statement)
            if not hits:
                continue
            examined += 1
            totals.update(hits)
            best_column[max(hits, key=lambda c: hits[c])] += 1

    print(f"\n{label}: {examined} statement tables with at least one match")
    print("  total alias matches per column:")
    for column in sorted(totals):
        print(f"    column {column}: {totals[column]:6d}")
    print("  table's best column:")
    for column, count in best_column.most_common():
        print(f"    column {column}: {count:5d} tables "
              f"({100 * count / max(examined, 1):5.1f}%)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel = build_panel(store.frame)
    frame = store.frame

    in_panel = set(panel)
    groups: dict[tuple, list] = collections.defaultdict(list)
    for row in frame.itertuples():
        if not bool(getattr(row, "eligible", True)):
            continue
        groups[(str(row.ticker), str(row.year), str(row.scope))].append(row)

    silent = [k for k in groups if k not in in_panel]
    served = [k for k in groups if k in in_panel]
    survey(silent, groups, store, "silent groups", 60)
    survey(served, groups, store, "served groups (control)", 60)


if __name__ == "__main__":
    main()
