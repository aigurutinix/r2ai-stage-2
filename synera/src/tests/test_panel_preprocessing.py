from __future__ import annotations

import unittest

from road2ai_vifinqa.build_panel import (
    _classify,
    _find_code,
    _is_current_period_source,
    _numeric_values,
)
from road2ai_vifinqa.table_semantics import TableAnalyzer


class PanelPreprocessingTest(unittest.TestCase):
    def test_finds_consistent_code_column_after_ordinal(self) -> None:
        rows = [
            ["STT", "Mã", "Chỉ tiêu", "2024", "2023"],
            ["1", "01", "Doanh thu bán hàng", "1.000", "900"],
            ["2", "11", "Giá vốn hàng bán", "700", "600"],
            ["3", "20", "Lợi nhuận gộp", "300", "300"],
            ["4", "25", "Chi phí bán hàng", "50", "40"],
            ["5", "26", "Chi phí quản lý", "30", "20"],
        ]
        result = _classify(rows, "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "kqkd")
        self.assertEqual(result.code_col, 1)
        self.assertEqual(_find_code(rows[1], code_col=result.code_col), (1, "01"))

    def test_numeric_values_skip_codes_years_and_small_ordinals(self) -> None:
        row = ["1", "01", "Doanh thu", "2024", "1.234.567"]
        self.assertEqual(_numeric_values(row, 1), [(4, 1234567.0, "1.234.567")])

    def test_rejects_prior_year_cell_when_current_value_is_blank(self) -> None:
        rows = [
            ["Chỉ tiêu", "2024", "2023"],
            ["Doanh thu", "", "1.000"],
        ]
        semantics = TableAnalyzer(rows, report_year=2024).cell(1, 2)
        self.assertFalse(_is_current_period_source(semantics, 2024))

    def test_small_amounts_are_not_note_numbers_in_period_columns(self) -> None:
        rows = [["Chỉ tiêu", "Mã", "Thuyết minh", "31/12/2024", "1/1/2024"],
                ["Tài sản khác", "150", "5", "125", "2024"]]
        analyzer = TableAnalyzer(rows, context="Đơn vị tính: triệu VND", report_year=2024)
        self.assertEqual(_numeric_values(rows[1], 1, analyzer=analyzer, row_idx=1),
                         [(3, 125.0, "125"), (4, 2024.0, "2024")])

    def test_reported_nil_is_zero_but_blank_cell_is_unknown(self) -> None:
        rows = [["Chỉ tiêu", "Mã", "Thuyết minh", "31/12/2024", "1/1/2024"],
                ["Tài sản khác", "150", "-", "—", ""]]
        analyzer = TableAnalyzer(rows, report_year=2024)
        self.assertEqual(_numeric_values(rows[1], 1, analyzer=analyzer, row_idx=1),
                         [(3, 0.0, "—")])

    def test_numeric_column_ordinals_are_not_statement_amounts(self) -> None:
        rows = [["Chỉ tiêu", "Mã", "Thuyết minh", "Năm 2024", "Năm 2023"],
                ["1", "2", "3", "4", "5"]]
        analyzer = TableAnalyzer(rows, context="Đơn vị tính: triệu VND", report_year=2024)
        self.assertEqual(_numeric_values(rows[1], 1, analyzer=analyzer, row_idx=1), [])


if __name__ == "__main__":
    unittest.main()
