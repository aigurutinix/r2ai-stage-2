"""Tests for structured selection (model picks cells, we write the pandas).

Motivation measured on submission #12 (Qwen 7B writing pandas itself):
35% of queries had no column filter, 15% skipped the unit conversion, 90%
omitted regex=False — 175 new answers moved the leaderboard by ~2 questions.
These tests pin that the synthesiser cannot reproduce any of those mistakes.
"""
import unittest

from vifinqa.codegen.selection import (Selection, parse_selection, synthesize,
                                       confidence, requirement_coverage,
                                       selection_matches_route)


class _C:
    """Minimal stand-in for retrieval.shortlist.Candidate."""

    def __init__(self, var="df1", label="Doanh thu thuần", col=1,
                 col_name="2023", value=100.0, unit_scale=1e6, score=80.0,
                 row=1, report_id="AAA_2023_consolidated", code="10"):
        self.var, self.label, self.col, self.col_name = var, label, col, col_name
        self.value, self.unit_scale, self.score = value, unit_scale, score
        self.row = row
        self.report_id, self.table_pos, self.code = report_id, 1, code


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        s = parse_selection('{"op": "lookup", "operands": [2]}')
        self.assertEqual((s.op, s.operands), ("lookup", [2]))

    def test_fenced_json_with_prose(self):
        s = parse_selection('Sure!\n```json\n{"op":"difference","operands":[1,3]}\n```\nDone')
        self.assertEqual((s.op, s.operands), ("difference", [1, 3]))

    def test_scalar_operand_is_wrapped(self):
        self.assertEqual(parse_selection('{"op":"lookup","operands":1}').operands, [1])

    def test_alternative_keys(self):
        self.assertEqual(parse_selection('{"op":"lookup","idx":[4]}').operands, [4])

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_selection("I cannot answer this question."))
        self.assertIsNone(parse_selection(""))


class ValidationTests(unittest.TestCase):
    def test_rejects_out_of_range_operand(self):
        self.assertIn("out of range", Selection("lookup", [9]).valid_for(3))

    def test_rejects_unknown_op(self):
        self.assertIn("unknown op", Selection("frobnicate", [1]).valid_for(3))

    def test_rejects_too_few_operands(self):
        self.assertIn("needs 2", Selection("difference", [1]).valid_for(3))


class RequirementCoverageTests(unittest.TestCase):
    @staticmethod
    def _requirement(ticker, requirement_id):
        return {
            "requirement_id": requirement_id,
            "ticker": ticker,
            "year": 2023,
            "doc_type": "consolidated",
            "metric_key": "net_revenue",
        }

    def test_complete_only_when_every_entity_metric_year_is_selected(self):
        candidates = [
            _C(var="df1", report_id="AAA_2023_consolidated"),
            _C(var="df2", report_id="BBB_2023_consolidated"),
        ]
        requirements = [self._requirement("AAA", "AAA|2023|net_revenue"),
                        self._requirement("BBB", "BBB|2023|net_revenue")]
        complete = requirement_coverage(
            Selection("average", [1, 2]), candidates, requirements)
        missing = requirement_coverage(
            Selection("lookup", [1]), candidates, requirements)
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["covered"], 2)
        self.assertFalse(missing["complete"])
        self.assertEqual(missing["missing"], ["BBB|2023|net_revenue"])

    def test_wrong_metric_or_year_does_not_prove_requirement(self):
        requirements = [self._requirement("AAA", "AAA|2023|net_revenue")]
        wrong_metric = _C(label="Tổng tài sản", code="270")
        wrong_year = _C(col_name="2022", report_id="AAA_2022_consolidated")
        self.assertFalse(requirement_coverage(
            Selection("lookup", [1]), [wrong_metric], requirements)["complete"])
        self.assertFalse(requirement_coverage(
            Selection("lookup", [1]), [wrong_year], requirements)["complete"])

    def test_no_structured_requirements_keeps_legacy_selection_available(self):
        state = requirement_coverage(Selection("lookup", [1]), [_C()], [])
        self.assertEqual(state, {"required": 0, "covered": 0,
                                 "complete": True, "missing": []})

    def test_generic_parent_does_not_prove_named_counterparty_requirement(self):
        requirement = {
            "requirement_id": "HNG|2017|borrowings_long_term",
            "ticker": "HNG", "year": 2017, "doc_type": "separate",
            "metric_key": "borrowings_long_term",
            "metric_variants": ["vay dai han voi hoang anh gia lai", "vay dai han"],
        }
        parent = _C(label="Vay dai han", code="", col_name="2017",
                    report_id="HNG_2017_separate")
        child = _C(
            label="Cong ty Co phan HoangAnh Gia Lai Cong ty me Vay dai han",
            code="", col_name="2017", report_id="HNG_2017_separate")

        self.assertFalse(requirement_coverage(
            Selection("lookup", [1]), [parent], [requirement])["complete"])
        self.assertTrue(requirement_coverage(
            Selection("lookup", [1]), [child], [requirement])["complete"])


class RouteCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _requirements(n=1, metric="net_revenue"):
        return [{
            "requirement_id": f"AAA|{2020 + i}|{metric}",
            "ticker": "AAA", "year": 2020 + i,
            "doc_type": "consolidated", "metric_key": metric,
        } for i in range(n)]

    def test_direct_lookup_requires_exactly_one_fact(self):
        route = {"question": "Doanh thu nam 2024?", "output_type": "number",
                 "plan": {"op": "lookup"}}
        self.assertTrue(selection_matches_route(
            Selection("lookup", [1]), route, self._requirements()))
        self.assertFalse(selection_matches_route(
            Selection("sum", [1, 2]), route, self._requirements(2)))

    def test_simple_ranking_accepts_one_metric_across_periods(self):
        route = {"question": "Doanh thu lon nhat trong cac nam la bao nhieu?",
                 "output_type": "number", "plan": {"op": "ranking"}}
        self.assertTrue(selection_matches_route(
            Selection("ranking_max", [1, 2, 3]), route,
            self._requirements(3)))

    def test_nested_ranking_is_not_representable_by_one_selection_op(self):
        route = {
            "question": "Doanh thu tai nam co von chu so huu cao nhat la bao nhieu?",
            "output_type": "number", "plan": {"op": "ranking"},
        }
        self.assertFalse(selection_matches_route(
            Selection("ranking_max", [1, 2, 3]), route,
            self._requirements(3)))

    def test_conditional_count_is_rejected(self):
        route = {"question": "Co bao nhieu cong ty co doanh thu duong?",
                 "output_type": "count", "plan": {"op": "count"}}
        self.assertFalse(selection_matches_route(
            Selection("count", [1, 2]), route, self._requirements(3)))


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.route = {"unit_scale": 1e9, "output_type": "number"}

    def test_lookup_applies_unit_conversion(self):
        # cell is 100 (triệu VND); question asks tỷ đồng -> 100*1e6/1e9 = 0.1
        c = _C(value=100.0, unit_scale=1e6)
        ans, q, err = synthesize(Selection("lookup", [1]), [c], self.route)
        self.assertIsNone(err)
        self.assertAlmostEqual(ans, 0.1, places=6)
        self.assertIn("/ 1e+09", q)

    def test_generated_query_always_pins_column_and_exact_label(self):
        c = _C(label="Doanh thu (thuần)", col=3, row=7)
        _a, q, _e = synthesize(Selection("lookup", [1]), [c], self.route)
        self.assertIn("['row'] == 7", q)
        self.assertIn(".str.strip().eq(", q)
        self.assertNotIn("str.contains", q)
        self.assertIn("['col'] == 3", q)
        compile(q, "<q>", "eval")          # single expression, grader-compatible

    def test_growth_is_percent_and_ordered_end_first(self):
        end, base = _C(value=120.0, unit_scale=1.0), _C(value=100.0, unit_scale=1.0)
        ans, _q, err = synthesize(Selection("growth_pct", [1, 2]), [end, base],
                                  {"unit_scale": 1.0, "output_type": "percent"})
        self.assertIsNone(err)
        self.assertAlmostEqual(ans, 20.0)

    def test_ratio_returns_percent_not_fraction(self):
        num, den = _C(value=9.0, unit_scale=1.0), _C(value=100.0, unit_scale=1.0)
        ans, _q, _e = synthesize(Selection("ratio", [1, 2]), [num, den],
                                 {"unit_scale": 1.0, "output_type": "percent"})
        self.assertAlmostEqual(ans, 9.0)

    def test_ratio_times_is_not_scaled(self):
        a, b = _C(value=9.0, unit_scale=1.0), _C(value=3.0, unit_scale=1.0)
        ans, _q, _e = synthesize(Selection("ratio_times", [1, 2]), [a, b],
                                 {"unit_scale": 1.0, "output_type": "ratio"})
        self.assertAlmostEqual(ans, 3.0)

    def test_ranking_max_and_min(self):
        cs = [_C(var="df1", value=1.0, unit_scale=1e9),
              _C(var="df2", value=7.0, unit_scale=1e9),
              _C(var="df3", value=4.0, unit_scale=1e9)]
        hi, _q, _e = synthesize(Selection("ranking_max", [1, 2, 3]), cs, self.route)
        lo, _q2, _e2 = synthesize(Selection("ranking_min", [1, 2, 3]), cs, self.route)
        self.assertAlmostEqual(hi, 7.0)
        self.assertAlmostEqual(lo, 1.0)

    def test_average_and_sum(self):
        cs = [_C(var="df1", value=2.0, unit_scale=1e9),
              _C(var="df2", value=4.0, unit_scale=1e9)]
        s, _q, _e = synthesize(Selection("sum", [1, 2]), cs, self.route)
        a, _q2, _e2 = synthesize(Selection("average", [1, 2]), cs, self.route)
        self.assertAlmostEqual(s, 6.0)
        self.assertAlmostEqual(a, 3.0)

    def test_count_is_dataframe_grounded(self):
        cs = [_C(var="df1", value=2.0, unit_scale=1.0),
              _C(var="df2", value=4.0, unit_scale=1.0),
              _C(var="df3", value=-1.0, unit_scale=1.0)]
        ans, q, err = synthesize(Selection("count", [1, 3]), cs,
                                 {"unit_scale": 1.0, "output_type": "count"})
        self.assertIsNone(err)
        self.assertAlmostEqual(ans, 2.0)
        self.assertIn("df1", q)
        self.assertIn("df3", q)
        compile(q, "<q>", "eval")

    def test_division_by_zero_is_reported_not_raised(self):
        num, den = _C(value=1.0, unit_scale=1.0), _C(value=0.0, unit_scale=1.0)
        ans, _q, err = synthesize(Selection("ratio", [1, 2]), [num, den], self.route)
        self.assertIsNone(ans)
        self.assertIn("zero", err)

    def test_percent_lookup_scales_a_ratio_cell(self):
        c = _C(label="Tỷ lệ sở hữu", value=0.9, unit_scale=1.0)
        ans, q, _e = synthesize(Selection("lookup", [1]), [c],
                                {"unit_scale": 1.0, "output_type": "percent"})
        self.assertAlmostEqual(ans, 90.0)      # organizer-confirmed: 90 not 0.9
        self.assertIn("* 100", q)


class ConfidenceTests(unittest.TestCase):
    def test_confidence_follows_weakest_pick(self):
        cs = [_C(score=90.0), _C(score=64.0)]
        c = confidence(Selection("difference", [1, 2]), cs, 10.0,
                       {"output_type": "number"})
        self.assertAlmostEqual(c, 64.0)

    def test_implausible_unit_lowers_confidence(self):
        cs = [_C(score=90.0)]
        good = confidence(Selection("lookup", [1]), cs, 90.0, {"output_type": "percent"})
        bad = confidence(Selection("lookup", [1]), cs, 0.9, {"output_type": "percent"})
        self.assertLess(bad, good)


if __name__ == "__main__":
    unittest.main()
