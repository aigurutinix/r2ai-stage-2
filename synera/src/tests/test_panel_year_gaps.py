from __future__ import annotations

import unittest

import pandas as pd

from road2ai_vifinqa.comparative_panel import is_prior_annual_column
from road2ai_vifinqa.panel import RAW_COLUMNS, enrich_panel


class PanelYearGapTest(unittest.TestCase):
    def test_rolling_metrics_do_not_skip_missing_years(self):
        rows = [{"ticker": ticker, "year": year, **{name: value for name in RAW_COLUMNS}}
                for ticker, year, value in [("AAA", 2021, 100.0), ("AAA", 2024, 200.0), ("AAA", 2025, 300.0), ("BBB", 2025, 400.0)]]
        frame = enrich_panel(pd.DataFrame(rows))
        rolling = ["roa", "roe", "inventory_days", "asset_turnover_avg", "equity_multiplier",
                   "operating_accruals_ratio", "revenue_growth", "gross_margin_change", "dol"]
        self.assertTrue(frame.loc[(frame.ticker == "AAA") & (frame.year == 2024), rolling].isna().all().all())
        self.assertEqual(float(frame.loc[(frame.ticker == "AAA") & (frame.year == 2025), "revenue_growth"].iloc[0]), 50.0)
        self.assertTrue(frame.loc[frame.ticker == "BBB", rolling].isna().all().all())

    def test_only_unambiguous_annual_comparatives(self):
        for text in ("Năm 2023", "31/12/2023", "Năm trước"):
            self.assertTrue(is_prior_annual_column(text, report_year=2024, balance=False))
        for text in ("Năm 2024", "Năm 2022", "Kỳ trước", "Số đầu kỳ", "Thuyết minh"):
            self.assertFalse(is_prior_annual_column(text, report_year=2024, balance=True))
        self.assertTrue(is_prior_annual_column("1/1/2024", report_year=2024, balance=True))
        self.assertFalse(is_prior_annual_column("1/1/2024", report_year=2024, balance=False))
        self.assertFalse(is_prior_annual_column("1/1/2023", report_year=2024, balance=True))


if __name__ == "__main__":
    unittest.main()
