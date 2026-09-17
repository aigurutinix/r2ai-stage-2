"""How many panel holes the accounting identities could fill on their own.

The panel is 39% full on average while 78% of its groups reached a balance sheet,
so the lines are there and the extraction missed them. Where two terms of an
identity survived, the third is not missing at all — it is one subtraction away,
and the identities were verified to hold on every group that could be checked.

This only measures the headroom. Nothing is written.

Usage:  PYTHONPATH=src python scripts/_panel_identities.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.corpus.metrics import build_panel  # noqa: E402
from vifin.store import TableStore  # noqa: E402

# (result, left, right) meaning result = left + right, applied in every direction.
IDENTITIES: tuple[tuple[str, str, str], ...] = (
    ("total_assets", "liabilities", "equity"),
    ("total_assets", "current_assets", "long_assets"),
    ("liabilities", "liabilities_short", "liabilities_long"),
    ("net_revenue", "gross_profit", "cogs"),
)


def solve(row: dict, filled: collections.Counter) -> int:
    """Fill what the identities determine, repeatedly, until nothing new appears."""

    gained = 0
    for _ in range(4):  # a fixpoint: one fill can complete another identity
        before = gained
        for whole, left, right in IDENTITIES:
            w, a, b = row.get(whole), row.get(left), row.get(right)
            if w is None and a is not None and b is not None:
                row[whole] = a + b
                filled[whole] += 1
                gained += 1
            elif a is None and w is not None and b is not None:
                row[left] = w - b
                filled[left] += 1
                gained += 1
            elif b is None and w is not None and a is not None:
                row[right] = w - a
                filled[right] += 1
                gained += 1
        if gained == before:
            break
    return gained


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, _ = build_panel(store.frame, with_provenance=True)

    involved = {name for triple in IDENTITIES for name in triple}
    before = sum(1 for row in panel.values() for m in involved if row.get(m) is not None)

    filled: collections.Counter[str] = collections.Counter()
    groups_helped = 0
    # Check the identities where both sides already exist, so a claim that they
    # hold is measured here rather than inherited from another script.
    agree = disagree = 0
    for row in panel.values():
        for whole, left, right in IDENTITIES:
            w, a, b = row.get(whole), row.get(left), row.get(right)
            if None in (w, a, b):
                continue
            scale = max(abs(w), 1.0)
            if abs(w - (a + b)) / scale <= 0.005:
                agree += 1
            else:
                disagree += 1
        if solve(row, filled):
            groups_helped += 1

    total_cells = len(panel) * len(involved)
    after = sum(1 for row in panel.values() for m in involved if row.get(m) is not None)
    print(f"{len(panel)} groups, {len(involved)} metrics touched by the identities")
    print(f"  checkable instances: {agree} hold, {disagree} do not "
          f"({100 * agree / max(agree + disagree, 1):.1f}% agreement)")
    print(f"  filled cells: {before} -> {after} of {total_cells} "
          f"({100 * before / total_cells:.1f}% -> {100 * after / total_cells:.1f}%)")
    print(f"  groups gaining at least one metric: {groups_helped} "
          f"({100 * groups_helped / len(panel):.1f}%)")
    print("\n  cells gained per metric:")
    for name, count in filled.most_common():
        print(f"    {count:5d}  {name}")


if __name__ == "__main__":
    main()
