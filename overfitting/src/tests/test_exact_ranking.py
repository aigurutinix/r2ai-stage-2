import io
import unittest

import pandas as pd

from vifinqa.codegen.exact_ranking import try_exact_ranking_answer


def _table(var, ticker, year, value):
    rows = [{
        "row": 3, "label": "Doanh thu thuần", "code": "10", "col": 3,
        "col_name": f"Năm {year}", "value": value, "unit_scale": 1e9,
    }]
    return {
        "var": var,
        "report_id": f"{ticker}_financial_statements_{year}_consolidated",
        "report_year": year, "table_pos": 5,
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
        "question": question, "output_type": "number", "unit_scale": 1e9,
        "plan": {"op": "ranking"},
        "evidence_requirements": requirements,
    }


def _eval(query, tables):
    frames = {
        table["var"]: pd.read_csv(io.StringIO(table["csv_text"]))
        for table in tables
    }
    scope = frames | {
        "round": round, "float": float, "abs": abs, "min": min, "max": max,
    }
    return eval(query, scope)


class ExactRankingTests(unittest.TestCase):
    def test_max_across_years_replays(self):
        requirements = [
            _requirement("AAA", 2022), _requirement("AAA", 2023),
            _requirement("AAA", 2024),
        ]
        tables = [
            _table("df1", "AAA", 2022, 80),
            _table("df2", "AAA", 2023, 125),
            _table("df3", "AAA", 2024, 110),
        ]
        answer = try_exact_ranking_answer(
            _route("Doanh thu cao nhất của AAA trong các năm là bao nhiêu?",
                   requirements),
            tables,
        )

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 125.0)
        self.assertEqual(answer.tier, "vas_ranking_current")
        self.assertEqual(_eval(answer.pandas_query, tables), 125.0)

    def test_min_across_entities(self):
        requirements = [
            _requirement("AAA", 2024), _requirement("BBB", 2024),
            _requirement("CCC", 2024),
        ]
        tables = [
            _table("df1", "AAA", 2024, 80),
            _table("df2", "BBB", 2024, 125),
            _table("df3", "CCC", 2024, 110),
        ]
        answer = try_exact_ranking_answer(
            _route("Doanh thu thấp nhất trong AAA, BBB và CCC?", requirements),
            tables,
        )

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 80.0)

    def test_recorded_level_does_not_match_credit_limit_qualifier(self):
        requirements = [
            _requirement("AAA", 2023), _requirement("AAA", 2024),
        ]
        tables = [
            _table("df1", "AAA", 2023, 80),
            _table("df2", "AAA", 2024, 125),
        ]
        answer = try_exact_ranking_answer(
            _route("AAA ghi nhận mức doanh thu cao nhất là bao nhiêu?", requirements),
            tables,
        )

        self.assertTrue(answer.ok, answer.detail)
        self.assertEqual(answer.answer, 125.0)

    def test_refuses_select_then_project(self):
        requirements = [
            _requirement("AAA", 2023), _requirement("AAA", 2024),
        ]
        answer = try_exact_ranking_answer(
            _route(
                "Lợi nhuận tại năm có doanh thu cao nhất của AAA là bao nhiêu?",
                requirements,
            ),
            [],
        )

        self.assertFalse(answer.ok)
        self.assertIn("nested ranking", answer.detail)

    def test_refuses_multiple_metrics(self):
        requirements = [
            _requirement("AAA", 2024),
            _requirement("BBB", 2024, "gross_profit"),
        ]
        answer = try_exact_ranking_answer(
            _route("Giá trị cao nhất giữa AAA và BBB?", requirements), [])

        self.assertFalse(answer.ok)
        self.assertIn("ranking metrics", answer.detail)


if __name__ == "__main__":
    unittest.main()
