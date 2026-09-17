import unittest
import pandas as pd

from road2ai_vifinqa.expression_plan import inline_plan
from road2ai_vifinqa.submission import evaluate_expression, _numpy_internal_import


class ExpressionPlanTest(unittest.TestCase):
    def test_short_steps_compile_to_one_equivalent_expression(self):
        frame = pd.DataFrame({"source_id": ["s1", "s2"], "value": [25.0, 100.0]})
        steps = [{"name": "x", "expression": "df.set_index('source_id')['value']"},
                 {"name": "ratio", "expression": "x.loc['s1'] / x.loc['s2']"}]
        query = inline_plan(steps, "float(ratio * 100)", frames={"df"}, columns=set(frame))
        self.assertEqual(evaluate_expression(query, {"df": frame}), 25.0)

    def test_forward_references_and_rebinding_are_rejected(self):
        for steps in ([{"name": "x", "expression": "y+1"}],
                      [{"name": "df", "expression": "df"}]):
            with self.assertRaises(ValueError):
                inline_plan(steps, "1", frames={"df"}, columns=set())

    def test_file_access_is_rejected_before_execution(self):
        with self.assertRaises(ValueError):
            inline_plan([], "pd.read_csv('outside.csv')", frames={"df"}, columns=set())

    def test_inplace_changes_are_rejected(self):
        with self.assertRaises(ValueError):
            inline_plan([], "df.dropna(inplace=True)", frames={"df"}, columns=set())

    def test_numpy_reduction_runs_without_general_import_access(self):
        frames={'df':pd.DataFrame({'value':[10.,30.]})}
        self.assertEqual(evaluate_expression("df['value'].values.mean()",frames),20.)
        with self.assertRaises(ImportError):_numpy_internal_import('os')
        with self.assertRaises(ValueError):evaluate_expression("__import__('os')",frames)
    def test_cartesian_merge_requires_cardinality_validation(self):
        with self.assertRaises(ValueError):inline_plan([],"df.merge(df, on='ticker')['value_x'].sum()",frames={'df'},columns={'ticker','value'})
        expression="df.merge(df, on='ticker', validate='one_to_one')['value_x'].sum()"
        frame=pd.DataFrame({'ticker':['A','B'],'value':[10.,20.]})
        query=inline_plan([],expression,frames={'df'},columns=set(frame))
        self.assertEqual(evaluate_expression(query,{'df':frame}),30.)


if __name__ == "__main__":
    unittest.main()
