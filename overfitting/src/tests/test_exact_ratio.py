import io
import unittest

import pandas as pd

from vifinqa.codegen.exact_ratio import try_exact_ratio_answer


def _requirement(metric_key, label, *, ticker="AAA", year=2024):
    return {
        "requirement_id": f"{ticker}|{year}|{metric_key}",
        "ticker": ticker, "year": year, "doc_type": "consolidated",
        "metric_key": metric_key, "metric_label": label,
        "metric_variants": [label], "statement": "income_statement",
    }


def _table(var, metric, code, value, scale, *, pos):
    rows = [{
        "row": 3, "label": metric, "code": code, "col": 3,
        "col_name": "Năm 2024", "value": value, "unit_scale": scale,
    }]
    return {
        "var": var,
        "report_id": "AAA_financial_statements_2024_consolidated",
        "report_year": 2024, "table_pos": pos,
        "context": "Báo cáo kết quả hoạt động kinh doanh",
        "grid_json": "[]", "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }


def _route(question, requirements, facts=None):
    if facts is None:
        facts = [
            {"role": "numerator", "metric": requirements[0]["metric_label"]},
            {"role": "denominator", "metric": requirements[-1]["metric_label"]},
        ]
    return {
        "question": question, "tickers": ["AAA"], "years": [2024],
        "doc_type": "consolidated", "output_type": "percent",
        "unit_scale": 1.0, "metric_keys": [
            requirement["metric_key"] for requirement in requirements],
        "plan": {"op": "ratio", "facts": facts},
        "evidence_requirements": requirements,
    }


def _eval(query, tables):
    frames = {
        table["var"]: pd.read_csv(io.StringIO(table["csv_text"]))
        for table in tables
    }
    return eval(query, frames | {"round": round, "float": float, "abs": abs})


class ExactRatioTests(unittest.TestCase):
    def test_direct_ratio_normalizes_mixed_units_and_replays(self):
        requirements = [
            _requirement("selling_expense", "chi phi ban hang"),
            _requirement("net_revenue", "doanh thu thuan"),
        ]
        tables = [
            _table("df1", "Chi phí bán hàng", "25", 120000, 1e6, pos=1),
            _table("df2", "Doanh thu thuần", "10", 2.0, 1e12, pos=2),
        ]

        answer = try_exact_ratio_answer(
            _route("Tỷ lệ chi phí bán hàng trên doanh thu thuần?", requirements),
            tables,
        )

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 6.0)
        self.assertEqual(_eval(answer.pandas_query, tables), 6.0)

    def test_expense_sign_is_normalized_to_absolute_value(self):
        requirements = [
            _requirement("administrative_expense", "chi phi quan ly doanh nghiep"),
            _requirement("net_revenue", "doanh thu thuan"),
        ]
        tables = [
            _table("df1", "Chi phí quản lý doanh nghiệp", "26", -20, 1e9, pos=1),
            _table("df2", "Doanh thu thuần", "10", 200, 1e9, pos=2),
        ]

        answer = try_exact_ratio_answer(
            _route("Tỷ lệ chi phí quản lý trên doanh thu thuần?", requirements),
            tables,
        )

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 10.0)
        self.assertEqual(_eval(answer.pandas_query, tables), 10.0)

    def test_grouped_selling_and_admin_expense_numerator(self):
        requirements = [
            _requirement("selling_expense", "chi phi ban hang"),
            _requirement("administrative_expense", "chi phi quan ly doanh nghiep"),
            _requirement("net_revenue", "doanh thu thuan"),
        ]
        tables = [
            _table("df1", "Chi phí bán hàng", "25", 10, 1e9, pos=1),
            _table("df2", "Chi phí quản lý doanh nghiệp", "26", 20, 1e9, pos=2),
            _table("df3", "Doanh thu thuần", "10", 200, 1e9, pos=3),
        ]
        facts = [
            {"role": "numerator", "metric": "chi phi ban hang va quan ly"},
            {"role": "denominator", "metric": "doanh thu thuan"},
        ]

        answer = try_exact_ratio_answer(
            _route("Tỷ lệ chi phí bán hàng và quản lý trên doanh thu?",
                   requirements, facts),
            tables,
        )

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 15.0)
        self.assertEqual(_eval(answer.pandas_query, tables), 15.0)

    def test_refuses_zero_denominator(self):
        requirements = [
            _requirement("selling_expense", "chi phi ban hang"),
            _requirement("net_revenue", "doanh thu thuan"),
        ]
        tables = [
            _table("df1", "Chi phí bán hàng", "25", 10, 1e9, pos=1),
            _table("df2", "Doanh thu thuần", "10", 0, 1e9, pos=2),
        ]

        answer = try_exact_ratio_answer(
            _route("Tỷ lệ chi phí bán hàng trên doanh thu?", requirements), tables)

        self.assertFalse(answer.ok)
        self.assertIn("denominator is zero", answer.detail)

    def test_refuses_opening_period(self):
        requirements = [
            _requirement("short_term_borrowings", "vay ngan han"),
            _requirement("liabilities", "tong no phai tra"),
        ]

        answer = try_exact_ratio_answer(
            _route("Tỷ lệ vay ngắn hạn trên nợ phải trả đầu năm 2024?",
                   requirements),
            [],
        )

        self.assertFalse(answer.ok)
        self.assertIn("opening-period", answer.detail)

    def test_refuses_gross_cost_mapped_to_net_tangible_assets(self):
        requirements = [
            _requirement("tangible_fixed_assets", "tai san co dinh huu hinh"),
            _requirement("total_assets", "tong tai san"),
        ]
        facts = [
            {"role": "numerator", "metric": "nguyen gia tai san co dinh huu hinh"},
            {"role": "denominator", "metric": "tong tai san"},
        ]

        answer = try_exact_ratio_answer(
            _route("Tỷ trọng nguyên giá tài sản cố định trên tổng tài sản?",
                   requirements, facts),
            [],
        )

        self.assertFalse(answer.ok)
        self.assertIn("role mismatch", answer.detail)

    def test_refuses_unsupported_three_operand_ratio(self):
        requirements = [
            _requirement("net_profit", "loi nhuan sau thue"),
            _requirement("gross_profit", "loi nhuan gop"),
            _requirement("net_revenue", "doanh thu thuan"),
        ]

        answer = try_exact_ratio_answer(
            _route("Tỷ lệ hai loại lợi nhuận trên doanh thu?", requirements), [])

        self.assertFalse(answer.ok)
        self.assertIn("unsupported grouped numerator", answer.detail)


if __name__ == "__main__":
    unittest.main()
