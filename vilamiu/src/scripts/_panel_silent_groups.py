"""Why 601 company-years in the corpus yield no metric at all.

Only 5 of 287 cohort questions hit a real corpus limit; 150 fail on groups that
are in `tables.parquet` and produce nothing. Those 601 silent groups are the
single largest lever left, so the question is what their tables look like: no
statement table at all, or statement tables whose row labels miss every alias.

Usage:  PYTHONPATH=src python scripts/_panel_silent_groups.py
"""

from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.corpus.metrics import METRICS, _fold, build_panel  # noqa: E402
from vifin.store import TableStore  # noqa: E402

ALIASES = {_fold(a) for m in METRICS for a in m.aliases}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, _ = build_panel(store.frame, with_provenance=True)
    frame = store.frame

    in_panel = set(panel)
    groups: dict[tuple, list] = collections.defaultdict(list)
    for row in frame.itertuples():
        groups[(str(row.ticker), str(row.year), str(row.scope))].append(row)

    silent = [key for key in groups if key not in in_panel]
    print(f"{len(silent)} silent groups of {len(groups)}\n")

    stat = collections.Counter()
    eligible_counts = []
    samples = []
    for key in silent:
        rows = groups[key]
        eligible = [r for r in rows if bool(getattr(r, "eligible", True))]
        eligible_counts.append(len(eligible))
        if not rows:
            stat["no table at all"] += 1
        elif not eligible:
            stat["tables exist but none eligible"] += 1
        else:
            stat["eligible tables, no alias matched"] += 1
            if len(samples) < 6:
                samples.append((key, eligible))

    for name, count in stat.most_common():
        print(f"  {count:4d}  {100 * count / len(silent):5.1f}%  {name}")
    eligible_counts.sort()
    if eligible_counts:
        mid = eligible_counts[len(eligible_counts) // 2]
        print(f"\n  median eligible tables in a silent group: {mid}")

    print("\nrow labels from silent groups that do have eligible tables:")
    rng = random.Random(11)
    for key, eligible in samples[:3]:
        pick = rng.choice(eligible)
        grid = store.rows(type(next(iter(panel)))(pick.doc_name, int(pick.table_id))) \
            if False else store.rows_by(pick.doc_name, int(pick.table_id)) \
            if hasattr(store, "rows_by") else None
        print(f"\n  {key}  doc={pick.doc_name} table={pick.table_id} "
              f"caption={str(getattr(pick, 'caption', ''))[:70]!r}")
        if grid:
            for line in grid[1:9]:
                label = str(line[0])[:60]
                mark = "MATCH" if _fold(label) in ALIASES else "     "
                print(f"      {mark}  {label}")


if __name__ == "__main__":
    main()
