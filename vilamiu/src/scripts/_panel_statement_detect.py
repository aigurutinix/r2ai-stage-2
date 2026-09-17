"""Do the silent groups have a main statement that detection misses?

`allow_label_only` is not the answer: the code already tried it and watched the
assets identity fall from 97.8% to 73.2%, because it lets a note table's "Tiền"
line stand in for the balance sheet's. The real gate is one line up — a table is
only treated as a statement when `detect_statement` names it and four of its rows
anchor to known aliases.

So the question is what detection sees in a silent group: no statement table at
all, or a statement table with too few anchored lines.

Usage:  PYTHONPATH=src python scripts/_panel_statement_detect.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.corpus.metrics import (  # noqa: E402
    BY_STATEMENT, MIN_STATEMENT_LINES, _fold, _matches_label, build_panel,
    detect_statement, find_code_column, value_columns,
)
from vifin.store import TableKey, TableStore  # noqa: E402


def anchored_lines(grid, statement) -> int:
    catalogue = BY_STATEMENT[statement]
    count = 0
    for row in grid[1:]:
        if not row:
            continue
        folded = _fold(row[0])
        if any(_matches_label(folded, a) for m in catalogue for a in m.aliases):
            count += 1
    return count


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
    outcome: collections.Counter[str] = collections.Counter()
    best_anchored: list[int] = []
    statements_seen: collections.Counter[str] = collections.Counter()

    for key in silent[:120]:
        detected = 0
        best = 0
        no_columns = 0
        too_short = 0
        for row in groups[key]:
            grid = store.rows(TableKey(row.doc_name, int(row.table_id)))
            if len(grid) < 3:
                too_short += 1
                continue
            statement = detect_statement(grid)
            if statement is None:
                continue
            detected += 1
            statements_seen[str(statement)] += 1
            code_column = find_code_column(grid)
            if not [c for c in value_columns(grid) if c != code_column]:
                no_columns += 1
                continue
            best = max(best, anchored_lines(grid, statement))
        best_anchored.append(best)
        if detected == 0:
            outcome["no table detected as a statement"] += 1
        elif no_columns == detected:
            outcome["statement detected, no usable value column"] += 1
        elif best < MIN_STATEMENT_LINES:
            outcome[f"statement detected, under {MIN_STATEMENT_LINES} anchored lines"] += 1
        else:
            outcome["should have produced metrics — look closer"] += 1

    sampled = sum(outcome.values())
    print(f"{sampled} silent groups sampled of {len(silent)}\n")
    for name, count in outcome.most_common():
        print(f"  {count:4d}  {100 * count / max(sampled, 1):5.1f}%  {name}")
    best_anchored.sort()
    if best_anchored:
        print(f"\n  best anchored-line count in a silent group: "
              f"median {best_anchored[len(best_anchored) // 2]}, "
              f"max {best_anchored[-1]}  (threshold is {MIN_STATEMENT_LINES})")
    print("\n  statements detected inside silent groups:")
    for name, count in statements_seen.most_common(8):
        print(f"    {count:5d}  {name}")


if __name__ == "__main__":
    main()
