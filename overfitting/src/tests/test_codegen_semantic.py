import unittest

from vifinqa.codegen.semantic import (
    all_dataframe_refs,
    answer_dataframe_refs,
    validate_generated_answer,
)
from vifinqa.codegen.generate import QuestionBundle


class SemanticValidationTests(unittest.TestCase):
    def test_expression_is_grounded(self):
        code = "round(float(df1['value'].iloc[0]), 2)"
        check = validate_generated_answer(code, {"df1"}, 12.34)
        self.assertTrue(check.ok)
        self.assertEqual(check.dataframe_refs, ["df1"])

    def test_follows_intermediate_dependencies(self):
        code = "x = float(df2['value'].iloc[0])\nanswer = x / 1e9"
        self.assertEqual(answer_dataframe_refs(code), {"df2"})

    def test_all_refs_ignore_comments_and_strings(self):
        code = "note = 'df3 is text'\n# df4 is a comment\nanswer = df1['value'] + df2['value']"
        self.assertEqual(all_dataframe_refs(code), {"df1", "df2"})

    def test_used_vars_include_executed_intermediate_dataframes(self):
        bundle = QuestionBundle.__new__(QuestionBundle)
        bundle.tables = [
            {"var": "df1", "report_id": "R1", "table_pos": 1},
            {"var": "df2", "report_id": "R2", "table_pos": 2},
        ]
        code = "tmp = df2['value'].sum()\nanswer = df1['value'].sum()"
        self.assertEqual(
            [item["var"] for item in bundle.used_vars(code)], ["df1", "df2"]
        )

    def test_rejects_constant_hallucination(self):
        check = validate_generated_answer("answer = 123.0", {"df1"}, 123.0)
        self.assertFalse(check.ok)
        self.assertIn("answer is not derived from any dataframe", check.errors)

    def test_dead_dataframe_reference_does_not_ground_constant(self):
        code = "unused = df1['value'].iloc[0]\nanswer = 123.0"
        check = validate_generated_answer(code, {"df1"}, 123.0)
        self.assertFalse(check.ok)
        self.assertTrue(check.warnings)

    def test_rejects_unknown_dataframe(self):
        check = validate_generated_answer("answer = float(df3.iloc[0, 0])", {"df1"}, 1)
        self.assertFalse(check.ok)
        self.assertIn("answer references unavailable dataframes: ['df3']", check.errors)

    def test_year_output_sanity(self):
        route = {"output_type": "year"}
        self.assertTrue(validate_generated_answer(
            "answer = float(df1.iloc[0, 0])", {"df1"}, 2022, route).ok)
        self.assertFalse(validate_generated_answer(
            "answer = float(df1.iloc[0, 0])", {"df1"}, 20.22, route).ok)


if __name__ == "__main__":
    unittest.main()
