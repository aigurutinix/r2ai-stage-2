"""Do the silent groups contain alias-matching row labels the panel never used?

599 of the 601 silent company-years hold a median of 52 eligible tables each and
still yield no metric. Fifty-two tables that never mention "tổng tài sản" is not a
wording problem, so the interesting question is whether the labels are there and
something downstream of matching drops them.

Usage:  PYTHONPATH=src python scripts/_panel_alias_scan.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.corpus.metrics import METRICS, _fold, build_panel  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

ALIAS_TO_METRIC = {_fold(a): m.name for m in METRICS for a in m.aliases}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, _ = build_panel(store.frame, with_provenance=True)
    frame = store.frame

    in_panel = set(panel)
    groups: dict[tuple, list] = collections.defaultdict(list)
    for row in frame.itertuples():
        if not bool(getattr(row, "eligible", True)):
            continue
        groups[(str(row.ticker), str(row.year), str(row.scope))].append(row)

    silent = [k for k in groups if k not in in_panel]
    print(f"{len(silent)} silent groups\n")

    found_any = 0
    per_group_hits = []
    metric_hits: collections.Counter[str] = collections.Counter()
    examples = []
    for key in silent[:120]:  # a sample: reading every grid of 601 groups is slow
        hits: collections.Counter[str] = collections.Counter()
        for row in groups[key]:
            grid = store.rows(TableKey(row.doc_name, int(row.table_id)))
            for line in grid:
                if not line:
                    continue
                metric = ALIAS_TO_METRIC.get(_fold(str(line[0])))
                if metric:
                    hits[metric] += 1
        per_group_hits.append(sum(hits.values()))
        if hits:
            found_any += 1
            metric_hits.update(hits)
            if len(examples) < 5:
                examples.append((key, len(groups[key]), dict(hits.most_common(4))))

    sampled = min(120, len(silent))
    print(f"of {sampled} sampled silent groups, {found_any} contain at least one "
          f"row label that maps to a metric ({100 * found_any / max(sampled, 1):.1f}%)")
    per_group_hits.sort()
    if per_group_hits:
        print(f"  median matching labels per silent group: "
              f"{per_group_hits[len(per_group_hits) // 2]}")
    print("\n  metrics whose labels are present but unused:")
    for name, count in metric_hits.most_common(10):
        print(f"    {count:5d}  {name}")
    print()
    for key, tables, hits in examples:
        print(f"  {key}  {tables} eligible tables  {hits}")


if __name__ == "__main__":
    main()
