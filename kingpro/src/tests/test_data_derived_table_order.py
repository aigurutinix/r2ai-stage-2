from __future__ import annotations

import unittest

from scripts.verify_data_derived_table_order import (
    _mrr5,
    _result_dependency_names,
)


class DataDerivedTableOrderTests(unittest.TestCase):
    def test_result_dependency_slice_keeps_selected_group(self) -> None:
        query = """
unused_ticker = 'AAA'
selected_ticker = frame.loc[frame.score.idxmax(), 'ticker']
answer_row = frame[frame.ticker == selected_ticker]
result = answer_row.value.iloc[0]
result = round(float(result), 2)
"""
        dependencies = _result_dependency_names(query)
        self.assertIn("selected_ticker", dependencies)
        self.assertIn("answer_row", dependencies)
        self.assertNotIn("unused_ticker", dependencies)

    def test_result_dependency_slice_follows_two_period_change(self) -> None:
        query = """
start, end = 2021, 2022
ticker = pivot[end].sub(pivot[start]).idxmax()
series = frame[frame.ticker == ticker].set_index('year').metric
result = series[end] - series[start]
"""
        dependencies = _result_dependency_names(query)
        self.assertTrue({"ticker", "start", "end"}.issubset(dependencies))

    def test_proxy_is_strictly_mrr_at_five(self) -> None:
        self.assertEqual(_mrr5(1), 1.0)
        self.assertEqual(_mrr5(5), 0.2)
        self.assertEqual(_mrr5(6), 0.0)


if __name__ == "__main__":
    unittest.main()
