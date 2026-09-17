import unittest
import pandas as pd
from road2ai_vifinqa.compliant_query import make_request,compile_plan,visible_frames,output_contract,extract_json
from road2ai_vifinqa.submission import evaluate_expression

class ComplianceBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.frames={'df':pd.DataFrame({'year':[2020,2021],'value':[100.,120.],
            'computed_answer':[999.,999.],'answer_value':[998.,998.],
            'question_id':[101,101],'retrieval_score':[100,100]})}
    def test_prompt_excludes_previous_answers_and_identifiers(self):
        prompt,key=make_request('Tăng trưởng năm 2021 là bao nhiêu %?',self.frames)
        self.assertNotIn('computed_answer',prompt)
        self.assertNotIn('answer_value',prompt)
        self.assertNotIn('question_id',prompt)
        changed={'df':self.frames['df'].assign(computed_answer=-50,answer_value=-60,question_id=700)}
        self.assertEqual((prompt,key),make_request('Tăng trưởng năm 2021 là bao nhiêu %?',changed))
    def test_calculates_growth_with_immutable_original_frames(self):
        plan={'steps':[{'name':'p','expression':"df.set_index('year')['value']"}],
              'expression':"float((p.loc[2021]/p.loc[2020]-1)*100)"}
        before=self.frames['df'].copy(deep=True)
        query,answer=compile_plan(plan,self.frames,'Tăng trưởng năm 2021 là bao nhiêu %?')
        self.assertAlmostEqual(answer,20)
        pd.testing.assert_frame_equal(before,self.frames['df'])
        self.assertNotIn('computed_answer',query)
        self.assertAlmostEqual(evaluate_expression(query,visible_frames(self.frames)),20)
        altered={'df':before.assign(value=[100.,150.])}
        self.assertAlmostEqual(evaluate_expression(query,altered),50)
    def test_rejects_precomputed_columns_and_answer_constants(self):
        for expression in ("float(df['computed_answer'].iloc[0])",'float(1)',
                           "float(df['value'].sum()*0+20)","float(df['value'].sum()+3.319357071984094)"):
            with self.subTest(expression=expression),self.assertRaises(ValueError):
                compile_plan({'expression':expression},self.frames,'Tính tăng trưởng')
    def test_all_frame_access_is_projected(self):
        query,answer=compile_plan({'expression':'float(df.iloc[0,-1])'},self.frames,'Lấy giá trị')
        self.assertEqual(answer,100.)
        self.assertNotIn('999',query)
    def test_query_cannot_read_files(self):
        with self.assertRaises(ValueError):
            compile_plan({'expression':"float(pd.read_csv('x')['value'].iloc[0])"},self.frames,'Lấy giá trị')

    def test_currency_conversion_is_explicit_and_question_driven(self):
        self.assertEqual(output_contract('Doanh thu là bao nhiêu tỷ đồng?')['compiler_divides_by'],1e9)
        self.assertEqual(output_contract('Bao nhiêu nghìn tỷ đồng?')['compiler_divides_by'],1e12)
        self.assertIsNone(output_contract('Doanh thu trên 50 tỷ đồng tăng bao nhiêu %?'))
        plan={'expression':"df['value'].iloc[0]",'expression_unit':'VND'}
        query,answer=compile_plan(plan,self.frames,'Bao nhiêu tỷ đồng?')
        self.assertAlmostEqual(answer,100/1e9)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            extract_json('{"expression":"df", "expression":"3"}')

    def test_raw_display_text_is_not_used_as_numeric_source(self):
        frames={'df':pd.DataFrame({'raw_value':[86.375],'raw_number':[86375.], 'source_scale':[1e6]})}
        with self.assertRaises(ValueError):
            compile_plan({'expression':"df['raw_value'].iloc[0]",'expression_unit':'VND'},frames,'Bao nhiêu triệu đồng?')
        _,answer=compile_plan({'expression':"(df['raw_number']*df['source_scale']).iloc[0]",'expression_unit':'VND'},frames,'Bao nhiêu triệu đồng?')
        self.assertEqual(answer,86375.)

if __name__=='__main__':unittest.main()
