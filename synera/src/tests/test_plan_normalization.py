import unittest
import pandas as pd
from road2ai_vifinqa.plan_normalization import normalized_json,vectorize_assign,normalize_display_conversion,expand_question_constants
from road2ai_vifinqa.compliant_query import compile_plan

class NormalizationTest(unittest.TestCase):
    def test_duplicate_steps_preserve_order_and_expression(self):
        parsed=normalized_json('{"steps":[{"name":"a","expression":"df[\"value\"]","name":"b","expression":"a.sum()"}],"expression":"b"}'.replace('df["value"]','df.value'))
        self.assertEqual([s['name'] for s in parsed['steps']],['a','b'])
    def test_ambiguous_duplicate_top_level_is_rejected(self):
        with self.assertRaises(ValueError):normalized_json('{"expression":"1","expression":"2"}')
    def test_assign_lowering_preserves_dependent_columns(self):
        source="df.assign(x=lambda d:d['value']*2,y=lambda d:d['x']+1)['y'].sum()"
        plan=vectorize_assign({'expression':source})
        self.assertNotIn('lambda',plan['expression'])
        self.assertEqual(compile_plan(plan,{'df':pd.DataFrame({'value':[5,6]})},'Tính tổng')[1],24)
    def test_outer_display_conversion_applied_once(self):
        frames={'df':pd.DataFrame({'value':[1234000000.]})}
        question='Doanh thu bao nhiêu tỷ đồng?'
        plan={'steps':[{'name':'p','expression':"df['value'].sum()"}],
              'expression':'float(p / 1000000000.0)','expression_unit':'VND'}
        revised=normalize_display_conversion(plan,question,frames)
        self.assertAlmostEqual(compile_plan(revised,frames,question)[1],1.234)
        rounded={**plan,'expression':'round(p / 1000000000.0, 2)'}
        self.assertEqual(normalize_display_conversion(rounded,question,frames),rounded)
    def test_pandas_threshold_mask_precedence(self):
        frames={'df':pd.DataFrame({'year':[2023,2024,2024],'ticker':['ABC','ABC','XYZ'],'value':[1.,2.,3.]})}
        for expression in ["df.loc[df['year']==2024 & (df['ticker']=='ABC'),'value'].sum()",
                           "df.loc[df['year']==2024 & df['ticker']=='ABC','value'].sum()"]:
            repaired=vectorize_assign({'expression':expression})
            self.assertEqual(compile_plan(repaired,frames,'Năm 2024')[1],2.)
    def test_expands_only_constants_grounded_in_question(self):
        frames={'df':pd.DataFrame({'value':[100.]})}
        question='Giảm 30% giá trị còn lại bao nhiêu?'
        plan=expand_question_constants({'expression':"df['value'].sum() * 0.7"},question)
        self.assertAlmostEqual(compile_plan(plan,frames,question)[1],70.)
        unrelated={'expression':"df['value'].sum() + 654321.0"}
        self.assertIn('654321.0',expand_question_constants(unrelated,question)['expression'])
        scale={'expression':"df['value'].sum() / 1000000000.0"}
        self.assertIn('1000000000.0',expand_question_constants(scale,'Có 1 tỷ đồng')['expression'])

if __name__=='__main__':unittest.main()
