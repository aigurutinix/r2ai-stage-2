"""Tests for the composite rule engine (fact resolver + formula executor +
rule/LLM arbitration).

Context: composite questions are ~50% of the test set and scored 0.000 with a
lookup-only engine. On the offline eval suite this module took the overall
answer accuracy from 0.157 to 0.397.
"""
import unittest

import pandas as pd

from vifinqa.codegen.arbitrate import agree, arbitrate
from vifinqa.codegen.fact_resolver import resolve_fact, _tables_for
from vifinqa.codegen.rule_composite import _FactView, try_composite_answer
from vifinqa.retrieval.shortlist import (_code_bonus, _year_status,
                                         build_shortlist)
from vifinqa.router.decompose import build_plan, split_ratio_metric


def _table(var, report_id, rows):
    return {"var": var, "report_id": report_id, "table_pos": 1,
            "report_year": int(report_id.split("_")[-2]),
            "csv_text": pd.DataFrame(rows).to_csv(index=False)}


def _row(label, value, code="", col=1, col_name="2015", unit=1e6, row=1):
    return {"row": row, "label": label, "code": code, "col": col,
            "col_name": col_name, "value": value, "unit_scale": unit}


class ScoringTests(unittest.TestCase):
    def test_year_status(self):
        self.assertEqual(_year_status("Năm 2015", [2015]), "match")
        self.assertEqual(_year_status("2014(theo báo cáo)", [2015]), "other")
        self.assertEqual(_year_status("Số cuối năm", [2015]), "none")

    def test_vas_code_separates_lookalike_labels(self):
        # "Lợi nhuận sau thuế" (60) vs "... chưa phân phối" (421)
        self.assertGreater(_code_bonus("60", ["loi nhuan sau thue"]),
                           _code_bonus("421", ["loi nhuan sau thue"]))

    def test_wrong_year_column_loses_to_right_one(self):
        tables = [_table("df1", "AAA_financial_statements_2015_consolidated", [
            _row("Doanh thu thuần", 100.0, "10", col=1, col_name="2014"),
            _row("Doanh thu thuần", 200.0, "10", col=2, col_name="2015", row=2),
        ])]
        cands = build_shortlist(tables, ["doanh thu thuan"], [2015])
        self.assertTrue(cands)
        self.assertEqual(cands[0].col_name, "2015")

    def test_current_period_beats_code_and_note_columns(self):
        tables = [_table("df1", "AAA_financial_statements_2024_consolidated", [
            _row("Doanh thu thuần", 3.0, "10", col=0, col_name="CHỈ TIÊU"),
            _row("Doanh thu thuần", 26.1, "10", col=2,
                 col_name="Thuyết minh", row=2),
            _row("Doanh thu thuần", 134_341.0, "10", col=3,
                 col_name="Năm nay", row=3),
            _row("Doanh thu thuần", 118_280.0, "10", col=4,
                 col_name="Năm trước", row=4),
        ])]
        cands = build_shortlist(tables, ["doanh thu thuan"], [2024])
        self.assertTrue(cands)
        self.assertEqual(cands[0].col_name, "Năm nay")
        self.assertEqual(cands[0].value, 134_341.0)

    def test_prior_period_used_from_next_year_report(self):
        tables = [_table("df1", "AAA_financial_statements_2025_consolidated", [
            _row("Doanh thu thuần", 150.0, "10", col=3,
                 col_name="Năm nay"),
            _row("Doanh thu thuần", 120.0, "10", col=4,
                 col_name="Năm trước", row=2),
        ])]
        cands = build_shortlist(tables, ["doanh thu thuan"], [2024])
        self.assertTrue(cands)
        self.assertEqual(cands[0].col_name, "Năm trước")
        self.assertEqual(cands[0].value, 120.0)


class FactResolverTests(unittest.TestCase):
    def setUp(self):
        self.tables = [
            _table("df1", "AAA_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 100.0, "10")]),
            _table("df2", "BBB_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 250.0, "10")]),
        ]

    def test_scopes_tables_by_ticker(self):
        scoped = _tables_for(self.tables, "BBB", 2015)
        self.assertEqual([t["var"] for t in scoped], ["df2"])

    def test_resolves_each_company_to_its_own_cell(self):
        a = resolve_fact(_FactView({"ticker": "AAA", "year": 2015, "metric": "doanh thu thuan"}),
                         self.tables, ["doanh thu thuan"])
        b = resolve_fact(_FactView({"ticker": "BBB", "year": 2015, "metric": "doanh thu thuan"}),
                         self.tables, ["doanh thu thuan"])
        self.assertEqual((a.var, b.var), ("df1", "df2"))
        self.assertAlmostEqual(a.value_vnd, 100.0 * 1e6)
        self.assertAlmostEqual(b.value_vnd, 250.0 * 1e6)
        self.assertIn("['row'] == 1", a.expr())
        self.assertIn(".str.strip().eq(", a.expr())
        self.assertNotIn("str.contains", a.expr())

    def test_unknown_ticker_yields_nothing(self):
        self.assertIsNone(resolve_fact(
            _FactView({"ticker": "ZZZ", "year": 2015, "metric": "doanh thu thuan"}),
            self.tables, ["doanh thu thuan"]))


class CompositeTests(unittest.TestCase):
    def _route(self, op, facts, **kw):
        r = {"metric_norm": kw.get("metric", "doanh thu thuan"),
             "metric_variants": [kw.get("metric", "doanh thu thuan")],
             "unit_scale": kw.get("unit_scale", 1e9),
             "output_type": kw.get("output_type", "number"),
             "question": kw.get("question", ""),
             "plan": {"op": op, "facts": facts}}
        return r

    def test_difference_between_two_companies(self):
        tables = [
            _table("df1", "AAA_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 5000.0, "10", unit=1e9)]),
            _table("df2", "BBB_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 2000.0, "10", unit=1e9)]),
        ]
        facts = [{"ticker": "AAA", "year": 2015, "metric": "doanh thu thuan"},
                 {"ticker": "BBB", "year": 2015, "metric": "doanh thu thuan"}]
        ca = try_composite_answer(self._route("difference", facts), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 3000.0, places=2)
        # the query must stay ONE expression (grader uses eval)
        compile(ca.pandas_query, "<q>", "eval")

    def test_chenh_lech_without_direction_returns_absolute_gap(self):
        tables = [
            _table("df1", "AAA_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 2.0, "10", unit=1e9)]),
            _table("df2", "BBB_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 5.0, "10", unit=1e9)]),
        ]
        facts = [
            {"ticker": "AAA", "year": 2015, "metric": "doanh thu thuan"},
            {"ticker": "BBB", "year": 2015, "metric": "doanh thu thuan"},
        ]
        ca = try_composite_answer(self._route(
            "difference", facts,
            question="Chênh lệch doanh thu giữa AAA và BBB là bao nhiêu?",
        ), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 3.0, places=2)
        self.assertIn("abs(", ca.pandas_query)

    def test_growth_uses_the_later_year_as_end(self):
        tables = [
            _table("df1", "AAA_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 120.0, "10", col_name="2015")]),
            _table("df2", "AAA_financial_statements_2014_consolidated",
                   [_row("Doanh thu thuần", 100.0, "10", col_name="2014")]),
        ]
        facts = [{"ticker": "AAA", "year": 2015, "metric": "doanh thu thuan"},
                 {"ticker": "AAA", "year": 2014, "metric": "doanh thu thuan"}]
        ca = try_composite_answer(
            self._route("growth_pct", facts, output_type="percent"), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 20.0, places=2)   # percent, not 0.2

    def test_decrease_question_returns_positive_magnitude(self):
        tables = [
            _table("df1", "AAA_financial_statements_2017_consolidated",
                   [_row("Chi phí lãi vay", 120.0, col_name="2017")]),
            _table("df2", "AAA_financial_statements_2018_consolidated",
                   [_row("Chi phí lãi vay", 90.0, col_name="2018")]),
        ]
        facts = [
            {"ticker": "AAA", "year": 2017, "metric": "chi phi lai vay"},
            {"ticker": "AAA", "year": 2018, "metric": "chi phi lai vay"},
        ]
        ca = try_composite_answer(self._route(
            "growth_pct", facts, output_type="percent",
            question="Chi phí lãi vay giảm bao nhiêu % từ 2017 đến 2018?",
            metric="chi phi lai vay",
        ), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 25.0, places=2)
        compile(ca.pandas_query, "<q>", "eval")

    def test_ratio_is_percent_not_fraction(self):
        tables = [_table("df1", "AAA_financial_statements_2015_consolidated", [
            _row("Lợi nhuận sau thuế", 9.0, "60", row=1),
            _row("Doanh thu thuần", 100.0, "10", row=2),
        ])]
        facts = [{"ticker": "AAA", "year": 2015, "metric": "loi nhuan sau thue",
                  "role": "numerator"},
                 {"ticker": "AAA", "year": 2015, "metric": "doanh thu thuan",
                  "role": "denominator"}]
        ca = try_composite_answer(
            self._route("ratio", facts, output_type="percent"), tables)
        self.assertTrue(ca.ok, ca.detail)
        self.assertAlmostEqual(ca.answer, 9.0, places=2)

    def test_ranking_picks_max_and_min(self):
        tables = [
            _table("df1", "AAA_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 1.0, "10", unit=1e9)]),
            _table("df2", "BBB_financial_statements_2015_consolidated",
                   [_row("Doanh thu thuần", 7.0, "10", unit=1e9)]),
        ]
        facts = [{"ticker": "AAA", "year": 2015, "metric": "doanh thu thuan"},
                 {"ticker": "BBB", "year": 2015, "metric": "doanh thu thuan"}]
        hi = try_composite_answer(self._route(
            "ranking", facts, question="cong ty nao co doanh thu lon nhat"), tables)
        lo = try_composite_answer(self._route(
            "ranking", facts, question="cong ty nao co doanh thu nho nhat"), tables)
        self.assertAlmostEqual(hi.answer, 7.0)
        self.assertAlmostEqual(lo.answer, 1.0)

    def test_unresolvable_fact_refuses_instead_of_guessing(self):
        tables = [_table("df1", "AAA_financial_statements_2015_consolidated",
                         [_row("Doanh thu thuần", 5.0, "10")])]
        facts = [{"ticker": "AAA", "year": 2015, "metric": "doanh thu thuan"},
                 {"ticker": "ZZZ", "year": 2015, "metric": "doanh thu thuan"}]
        ca = try_composite_answer(self._route("difference", facts), tables)
        self.assertFalse(ca.ok)


class DecomposeRatioTests(unittest.TestCase):
    def test_split_ratio_metric(self):
        self.assertEqual(split_ratio_metric("ty le loi nhuan sau thue tren doanh thu thuan"),
                         ("loi nhuan sau thue", "doanh thu thuan"))
        self.assertEqual(split_ratio_metric("doanh thu thuan"), ("", ""))

    def test_ratio_plan_has_two_metrics(self):
        plan = build_plan("Tỷ lệ lợi nhuận sau thuế trên doanh thu thuần của A năm 2015?",
                          ["AAA"], [2015], "consolidated",
                          "ty le loi nhuan sau thue tren doanh thu thuan")
        self.assertEqual(plan.op, "ratio")
        self.assertEqual({f.role for f in plan.facts}, {"numerator", "denominator"})

    def test_unsplittable_ratio_falls_back_to_lookup(self):
        plan = build_plan("Tỷ lệ nợ xấu của A năm 2015?", ["AAA"], [2015],
                          "consolidated", "ty le no xau")
        self.assertEqual(plan.op, "lookup")


class ArbitrationTests(unittest.TestCase):
    def _r(self, ans, conf):
        return {"answer": ans, "pandas_query": "1.0", "confidence": conf, "source": "rule"}

    def _l(self, ans, conf=90.0):
        return {"answer": ans, "pandas_query": "2.0", "confidence": conf, "source": "llm"}

    def test_agreement_boosts_confidence_and_keeps_rule_query(self):
        v = arbitrate(self._r(10.0, 65.0), self._l(10.0))
        self.assertEqual(v.used, "rule")
        self.assertGreaterEqual(v.confidence, 85.0)

    def test_confident_rule_wins_a_disagreement(self):
        v = arbitrate(self._r(10.0, 90.0), self._l(99.0))
        self.assertEqual((v.used, v.answer), ("rule", 10.0))

    def test_weak_rule_defers_to_llm(self):
        v = arbitrate(self._r(10.0, 40.0), self._l(99.0))
        self.assertEqual((v.used, v.answer), ("llm", 99.0))

    def test_single_side_is_taken(self):
        self.assertEqual(arbitrate(None, self._l(5.0)).used, "llm")
        self.assertEqual(arbitrate(self._r(5.0, 10.0), None).used, "rule")
        self.assertIsNone(arbitrate(None, None))

    def test_agree_tolerance(self):
        self.assertTrue(agree(100.0, 100.005))
        self.assertFalse(agree(100.0, 101.0))
        self.assertFalse(agree(float("nan"), float("nan")))


if __name__ == "__main__":
    unittest.main()
