"""Guard the leaderboard-confirmed rule: pandas_query MUST be one expression.

Submission #6 shipped 233 multi-line scripts; every one is a SyntaxError under
eval and was scored as a crash (EXEC 0.085 -> 0.0613 while ANSWER rose to
0.1047). These tests pin the inliner that prevents a repeat.
"""
import unittest

import pandas as pd

from vifinqa.codegen.to_expression import (InlineError, to_single_expression,
                                           try_to_expression)


def _eval(expr, **ns):
    return eval(compile(expr, "<q>", "eval"), {"pd": pd, **ns})  # noqa: S307


class InlineTests(unittest.TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame({
            "row": [0, 1, 2], "label": ["Doanh thu", "Chi phi", "Loi nhuan"],
            "code": ["10", "11", "60"], "col": [1, 1, 1],
            "col_name": ["2023", "2023", "2023"],
            "value": [1000.0, -400.0, 600.0], "unit_scale": [1e6, 1e6, 1e6]})

    def test_plain_script_is_inlined_and_keeps_value(self):
        code = ("rows = df1[df1['label'].str.contains('Doanh thu', na=False)]\n"
                "val = rows['value'].iloc[0]\n"
                "answer = round(float(val) * 1e6 / 1e9, 2)")
        expr = to_single_expression(code)
        self.assertNotIn("\n", expr)
        compile(expr, "<q>", "eval")          # grader semantics
        self.assertAlmostEqual(_eval(expr, df1=self.df1), 1.0, places=6)

    def test_already_expression_passthrough(self):
        # ast.unparse normalises literals (1e3 -> 1000.0); the VALUE is what matters
        code = "round(float(df1['value'].iloc[0]) / 1e3, 2)"
        expr = to_single_expression(code)
        self.assertNotIn("\n", expr)
        self.assertEqual(_eval(expr, df1=self.df1), _eval(code, df1=self.df1))

    def test_single_line_assignment(self):
        expr = to_single_expression("answer = float(df1['value'].iloc[2])")
        self.assertEqual(_eval(expr, df1=self.df1), 600.0)

    def test_trailing_bare_expression_is_the_result(self):
        expr = to_single_expression("x = df1['value']\nfloat(x.iloc[0])")
        self.assertEqual(_eval(expr, df1=self.df1), 1000.0)

    def test_negative_values_survive(self):
        code = ("r = df1[df1['code'].astype(str) == '11']\n"
                "answer = round(float(r['value'].iloc[0]) * 1e6 / 1e6, 2)")
        self.assertEqual(_eval(to_single_expression(code), df1=self.df1), -400.0)

    def test_rejects_control_flow(self):
        with self.assertRaises(InlineError):
            to_single_expression("if True:\n    answer = 1.0\nelse:\n    answer = 2.0")

    def test_rejects_reassignment(self):
        with self.assertRaises(InlineError):
            to_single_expression("answer = 1.0\nanswer = 2.0")

    def test_rejects_tuple_target(self):
        with self.assertRaises(InlineError):
            to_single_expression("a, b = 1, 2\nanswer = a + b")

    def test_rejects_loops_and_imports(self):
        for bad in ("for i in range(3):\n    answer = i",
                    "import os\nanswer = 1.0",
                    "def f():\n    return 1\nanswer = f()"):
            with self.assertRaises(InlineError):
                to_single_expression(bad)

    def test_try_variant_never_raises(self):
        expr, err = try_to_expression("while True:\n    pass")
        self.assertIsNotNone(err)
        self.assertTrue(expr)

    def test_operator_precedence_is_preserved(self):
        code = "s = 2 + 3\nanswer = s * 4"
        self.assertEqual(_eval(to_single_expression(code)), 20.0)


if __name__ == "__main__":
    unittest.main()
