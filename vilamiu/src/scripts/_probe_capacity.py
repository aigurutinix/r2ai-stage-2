"""Can the metric panel actually supply the four missing question classes?

The exam's cohort and count questions are written in the panel's own vocabulary —
vốn lưu động ròng, ROA, hệ số nợ/tổng tài sản, tài sản ngắn hạn/nợ ngắn hạn,
CFO/nợ ngắn hạn, trung vị, tỷ trọng — so generating them needs no LLM and cannot
get the answer wrong. What it does need is density: a cohort question over five
companies requires five companies sharing a year and scope with the metrics the
question names, and a year-over-year question requires the same company in two
years.

This counts the supply for each shape before any generator is written, because a
generator for a shape the corpus cannot fill is wasted work.
"""

from __future__ import annotations

import collections
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# The metric combinations the exam's cohort and screen questions actually use.
RATIOS: dict[str, tuple[str, str]] = {
    "tài sản ngắn hạn / nợ ngắn hạn": ("current_assets", "liabilities_short"),
    "nợ phải trả / tổng tài sản": ("liabilities", "total_assets"),
    "hàng tồn kho / nợ ngắn hạn": ("inventory", "liabilities_short"),
    "CFO / nợ ngắn hạn": ("cfo", "liabilities_short"),
    "ROA = LNST / tổng tài sản": ("net_profit", "total_assets"),
    "ROE = LNST / vốn chủ sở hữu": ("net_profit", "equity"),
    "D/E = nợ phải trả / vốn CSH": ("liabilities", "equity"),
    "biên lợi nhuận gộp": ("gross_profit", "net_revenue"),
    "CFO margin": ("cfo", "net_revenue"),
}

# Screens the exam states as a sign condition on a difference.
SCREENS: dict[str, tuple[str, str]] = {
    "vốn lưu động ròng (TSNH − NNH)": ("current_assets", "liabilities_short"),
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    panel = pd.read_parquet(ROOT / "artifacts" / "metrics.parquet")
    print(f"panel: {len(panel)} (ticker, year, scope) rows, "
          f"{panel['ticker'].nunique()} tickers, {panel['year'].nunique()} years\n")

    # --- shape 1 & 2: two-cell arithmetic and growth, same company two years ---
    pairs = 0
    for metric in ("net_revenue", "total_assets", "cfo", "net_profit",
                   "liabilities_short", "inventory", "equity"):
        if metric not in panel.columns:
            continue
        have = panel.dropna(subset=[metric])
        by_entity = have.groupby(["ticker", "scope"])["year"].nunique()
        n = int((by_entity >= 2).sum())
        pairs += n
        print(f"  {metric:<20} entities with >=2 years: {n}")
    print(f"\ntwo-year shapes available (entity x metric): {pairs}")

    # --- shape 3 & 4: cohort, several companies sharing a year and scope ------
    print("\ncohort supply per ratio "
          "(groups of >=4 companies sharing year+scope with both metrics):")
    for label, (num, den) in {**RATIOS, **SCREENS}.items():
        if num not in panel.columns or den not in panel.columns:
            print(f"  {label:<34} metric missing")
            continue
        have = panel.dropna(subset=[num, den])
        sizes = have.groupby(["year", "scope"])["ticker"].nunique()
        big = sizes[sizes >= 4]
        # A cohort question picks 4-7 of the companies available in that
        # (year, scope); the number of distinct questions is combinatorial, so
        # report the pool rather than a product that overstates usable variety.
        print(f"  {label:<34} {len(big):>3} year/scope groups, "
              f"{int(big.sum()):>4} company-slots, max group {int(sizes.max())}")

    # The binding constraint is the widest cohort we can pose. The exam poses
    # groups of 3 to 7, so report the distribution of achievable group sizes.
    have = panel.dropna(subset=["current_assets", "liabilities_short"])
    sizes = have.groupby(["year", "scope"])["ticker"].nunique()
    print("\nachievable cohort widths (TSNH + NNH present):",
          dict(sorted(collections.Counter(sizes.values).items())))


if __name__ == "__main__":
    main()
