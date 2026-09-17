from __future__ import annotations

import unittest

from road2ai_vifinqa.table_semantics import (
    PeriodRole,
    StatementKind,
    cell_semantics,
    column_header,
    period_info,
    statement_kind,
    unit_scale,
)


class TableSemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            ["Chỉ tiêu", "31/12/2024", "31/12/2023"],
            ["", "Số cuối năm", "Số đầu năm"],
            ["A. TÀI SẢN", "", ""],
            ["Tổng cộng tài sản", "1.000", "900"],
        ]

    def test_rebuilds_multirow_header_and_period(self) -> None:
        header = column_header(self.rows, 3, 1)
        self.assertIn("31/12/2024", header)
        self.assertIn("Số cuối năm", header)
        info = period_info(header, report_year=2024)
        self.assertEqual(info.year, 2024)
        self.assertEqual(info.role, PeriodRole.CLOSING)

    def test_cell_semantics_keeps_section_total_and_unit(self) -> None:
        semantics = cell_semantics(
            self.rows,
            3,
            1,
            context="BÁO CÁO TÌNH HÌNH TÀI CHÍNH. Đơn vị: triệu VND",
            report_year=2024,
        )
        self.assertIn("TÀI SẢN", semantics.section)
        self.assertTrue(semantics.is_total)
        self.assertEqual(semantics.unit_scale, 1_000_000.0)

    def test_table_local_vnd_beats_stale_context_unit(self) -> None:
        rows = [
            ["Đơn vị: VND", ""],
            ["Chỉ tiêu", "Năm 2024"],
            ["Tiền", "1.000"],
        ]
        self.assertEqual(
            unit_scale(rows, 2, 1, context="Trang trước dùng đơn vị: triệu VND"),
            1.0,
        )

    def test_statement_kind_uses_financial_statement_title(self) -> None:
        self.assertEqual(
            statement_kind("BÁO CÁO LƯU CHUYỂN TIỀN TỆ", self.rows),
            StatementKind.CASH_FLOW,
        )

    def test_opening_date_is_not_mistaken_for_current_year(self) -> None:
        info = period_info("Tại ngày 01/01/2024 (được báo cáo lại)", report_year=2024)
        self.assertEqual(info.year, 2024)
        self.assertEqual(info.role, PeriodRole.OPENING)

    def test_unpadded_opening_dates_are_recognized(self) -> None:
        for header in ("1/1/2024 VND", "01/1/2024", "1/01/2024"):
            with self.subTest(header=header):
                self.assertEqual(period_info(header, report_year=2024).role, PeriodRole.OPENING)


if __name__ == "__main__":
    unittest.main()
