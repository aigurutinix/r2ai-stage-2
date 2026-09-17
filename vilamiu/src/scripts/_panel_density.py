"""How full the metric panel actually is, per metric and per statement family.

92% of the cohort questions the panel refuses are refused because a metric is
missing from a group that exists. So the binding constraint is not which
company-years were parsed but which lines were found inside them, and that is
only actionable per metric.

Usage:  PYTHONPATH=src python scripts/_panel_density.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.corpus.metrics import METRICS, build_panel  # noqa: E402
from vifin.store import TableStore  # noqa: E402

# Which statement a line comes from, so a metric missing across a whole family
# reads as "that statement was never parsed" rather than "that alias is weak".
FAMILY = {
    "balance": {"cash", "receivables_short", "inventory", "current_assets",
                "long_assets", "total_assets", "liabilities", "liabilities_short",
                "liabilities_long", "equity", "loans_to_customers",
                "customer_deposits"},
    "income": {"net_revenue", "revenue_gross", "cogs", "gross_profit",
               "financial_income", "financial_expense", "interest_expense",
               "selling_expense", "admin_expense", "operating_profit",
               "profit_before_tax", "net_profit", "net_interest_income",
               "operating_income", "operating_expense"},
    "cashflow": {"cfo", "cfi", "cff"},
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, _ = build_panel(store.frame, with_provenance=True)
    groups = len(panel)
    print(f"{groups} company-year-scope groups\n")

    filled: collections.Counter[str] = collections.Counter()
    for row in panel.values():
        for name, value in row.items():
            if value is not None:
                filled[name] += 1

    names = [m.name for m in METRICS]
    print(f"{'metric':28s} {'groups':>7s} {'fill':>7s}")
    for name in sorted(names, key=lambda n: -filled[n]):
        print(f"  {name:26s} {filled[name]:7d} {100 * filled[name] / groups:6.1f}%")

    print("\nby statement family, mean fill across its metrics:")
    for family, members in FAMILY.items():
        present = [m for m in members if m in names]
        if not present:
            continue
        mean = sum(filled[m] for m in present) / (len(present) * groups)
        print(f"  {family:10s} {100 * mean:5.1f}%   ({len(present)} metrics)")

    whole = collections.Counter()
    for row in panel.values():
        for family, members in FAMILY.items():
            if any(row.get(m) is not None for m in members if m in names):
                whole[family] += 1
    print("\ngroups with at least one line from that statement:")
    for family in FAMILY:
        print(f"  {family:10s} {whole[family]:5d}  {100 * whole[family] / groups:5.1f}%")


if __name__ == "__main__":
    main()
