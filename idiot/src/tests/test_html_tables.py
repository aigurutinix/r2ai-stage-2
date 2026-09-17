from __future__ import annotations

import unittest

from road2ai_vifinqa.html_tables import parse_html_table


class HtmlTableExpansionTest(unittest.TestCase):
    def test_expands_rowspan_and_colspan_to_rectangular_grid(self) -> None:
        fragment = """
        <table>
          <tr><th rowspan="2">Chỉ tiêu</th><th colspan="2">Năm 2024</th></tr>
          <tr><th>Cuối năm</th><th>Đầu năm</th></tr>
          <tr><td>Tiền</td><td>1.000</td><td>900</td></tr>
        </table>
        """
        self.assertEqual(
            parse_html_table(fragment),
            [
                ["Chỉ tiêu", "Năm 2024", "Năm 2024"],
                ["Chỉ tiêu", "Cuối năm", "Đầu năm"],
                ["Tiền", "1.000", "900"],
            ],
        )

    def test_invalid_spans_fall_back_to_one(self) -> None:
        fragment = "<table><tr><td colspan='bad'>A</td><td rowspan='0'>B</td></tr></table>"
        self.assertEqual(parse_html_table(fragment), [["A", "B"]])


if __name__ == "__main__":
    unittest.main()
