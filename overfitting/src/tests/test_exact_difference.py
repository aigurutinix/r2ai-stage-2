import io
import unittest

import pandas as pd

from vifinqa.codegen.exact_difference import try_exact_difference_answer


def _table(var, ticker, year, value, *, row=3, metric="Doanh thu thuần", code="10"):
    rows = [{
        "row": row, "label": metric, "code": code, "col": 3,
        "col_name": f"Năm {year}", "value": value, "unit_scale": 1e9,
    }]
    return {
        "var": var,
        "report_id": f"{ticker}_financial_statements_{year}_consolidated",
        "report_year": year, "table_pos": row,
        "context": "Báo cáo kết quả hoạt động kinh doanh",
        "grid_json": "[]", "csv_text": pd.DataFrame(rows).to_csv(index=False),
    }


def _requirement(ticker, year, metric_key="net_revenue"):
    return {
        "requirement_id": f"{ticker}|{year}|{metric_key}",
        "ticker": ticker, "year": year, "doc_type": "consolidated",
        "metric_key": metric_key, "metric_label": "doanh thu thuan",
        "metric_variants": ["doanh thu thuan"],
        "statement": "income_statement",
    }


def _route(question, requirements):
    return {
        "question": question, "tickers": list(dict.fromkeys(
            requirement["ticker"] for requirement in requirements)),
        "years": list(dict.fromkeys(
            requirement["year"] for requirement in requirements)),
        "output_type": "number", "unit_scale": 1e9,
        "plan": {"op": "difference"},
        "evidence_requirements": requirements,
    }


def _eval(query, tables):
    frames = {
        table["var"]: pd.read_csv(io.StringIO(table["csv_text"]))
        for table in tables
    }
    return eval(query, frames | {"round": round, "float": float, "abs": abs})


class ExactDifferenceTests(unittest.TestCase):
    def test_two_entities_use_question_order_and_replay(self):
        requirements = [_requirement("AAA", 2024), _requirement("BBB", 2024)]
        tables = [_table("df1", "AAA", 2024, 120), _table("df2", "BBB", 2024, 70)]
        route = _route(
            "Doanh thu thuần của AAA chênh lệch bao nhiêu so với BBB?",
            requirements,
        )

        answer = try_exact_difference_answer(route, tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 50.0)
        self.assertEqual(answer.tier, "vas_pair_current")
        self.assertEqual(_eval(answer.pandas_query, tables), 50.0)

    def test_closing_date_is_not_mistaken_for_opening_date(self):
        requirements = [_requirement("AAA", 2024), _requirement("BBB", 2024)]
        tables = [_table("df1", "AAA", 2024, 120), _table("df2", "BBB", 2024, 70)]
        route = _route(
            "Chênh lệch doanh thu giữa AAA và BBB tại ngày 31/12/2024?",
            requirements,
        )

        answer = try_exact_difference_answer(route, tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 50.0)

    def test_two_periods_use_later_minus_earlier(self):
        requirements = [_requirement("AAA", 2020), _requirement("AAA", 2024)]
        tables = [_table("df1", "AAA", 2020, 80), _table("df2", "AAA", 2024, 125)]
        route = _route(
            "Mức thay đổi doanh thu thuần của AAA từ 2020 đến 2024?",
            requirements,
        )

        answer = try_exact_difference_answer(route, tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 45.0)
        self.assertEqual(_eval(answer.pandas_query, tables), 45.0)

    def test_lower_than_reverses_operands_to_return_gap(self):
        requirements = [_requirement("AAA", 2019), _requirement("AAA", 2017)]
        tables = [_table("df1", "AAA", 2019, 50), _table("df2", "AAA", 2017, 80)]
        route = _route(
            "Doanh thu năm 2019 thấp hơn năm 2017 bao nhiêu?", requirements)

        answer = try_exact_difference_answer(route, tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 30.0)

    def test_generic_difference_returns_absolute_gap(self):
        requirements = [_requirement("AAA", 2024), _requirement("BBB", 2024)]
        tables = [_table("df1", "AAA", 2024, 70), _table("df2", "BBB", 2024, 120)]
        route = _route(
            "Chênh lệch doanh thu thuần giữa AAA và BBB là bao nhiêu?",
            requirements,
        )

        answer = try_exact_difference_answer(route, tables)

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 50.0)
        self.assertEqual(_eval(answer.pandas_query, tables), 50.0)

    def test_refuses_child_qualifier_lost_by_router(self):
        requirements = [
            _requirement("AAA", 2023, "inventory"),
            _requirement("AAA", 2022, "inventory"),
        ]
        route = _route(
            "Chênh lệch giá gốc nguyên vật liệu của AAA giữa 2023 và 2022?",
            requirements,
        )

        answer = try_exact_difference_answer(route, [])

        self.assertFalse(answer.ok)
        self.assertIn("detail=nguyen vat lieu", answer.detail)

    def test_refuses_mismatched_metrics(self):
        requirements = [
            _requirement("AAA", 2024),
            _requirement("BBB", 2024, "gross_profit"),
        ]

        answer = try_exact_difference_answer(
            _route("Chênh lệch giữa AAA và BBB?", requirements), [])

        self.assertFalse(answer.ok)
        self.assertIn("difference metrics", answer.detail)


if __name__ == "__main__":
    unittest.main()
