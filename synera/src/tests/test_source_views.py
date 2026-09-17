import unittest
import pandas as pd
from road2ai_vifinqa.source_views import normalize_view
from road2ai_vifinqa.compliant_query import compile_plan,make_request
from road2ai_vifinqa.submission import evaluate_expression

class SourceViewTest(unittest.TestCase):
    def test_reconstructs_preweighted_value_from_raw(self):
        d=pd.DataFrame({'value':[10.,20.], 'raw_value':['40,00','80,00'],'source_scale':[1.,1.]})
        before=d.copy(deep=True)
        self.assertEqual(normalize_view('df',d)['value'].tolist(),[40.,80.])
        q,a=compile_plan({'expression':"df['value'].mean()"},{'df':d},'Trung bình là bao nhiêu %?')
        self.assertEqual(a,60.)
        self.assertEqual(evaluate_expression(q,{'df':d.assign(value=-999)}),60.)
        self.assertEqual(evaluate_expression(q,{'df':d.assign(raw_value=['60','80'])}),70.)
        pd.testing.assert_frame_equal(d,before)
    def test_grouping_signs_and_scale(self):
        d=pd.DataFrame({'value':[0.]*5,'raw_value':['1.234.567','1,234,567','(1.234,50)','99,91%','-'],'source_scale':[1e3,1,1,1,1]})
        self.assertEqual(normalize_view('df',d)['value'].tolist(),[1234567000.,1234567.,-1234.5,99.91,0.])
    def test_prompt_cannot_see_preweighted_value(self):
        d=pd.DataFrame({'value':[1.],'raw_value':['100'],'source_scale':[1.]})
        self.assertEqual(make_request('Tính giá trị',{'df':d}),make_request('Tính giá trị',{'df':d.assign(value=9.)}))
    def test_mixed_source_rows_keep_panel_values_and_recover_metadata(self):
        d=pd.DataFrame({'value':[10.,200.],'raw_value':['40',None],'raw_number':[40.,None],
            'source_scale':[1.,None],'raw':[None,'200'],'ticker':['AAA',None],
            'report_year':[2024.,None],'doc_id':['AAA_financial_statements_2024_consolidated']*2,
            'row_label':['Revenue',None],'label':[None,'Interest']})
        view=normalize_view('df',d)
        self.assertEqual(view['value'].tolist(),[40.,200.])
        self.assertEqual(view['ticker'].tolist(),['AAA','AAA'])
        self.assertEqual(view['report_year'].tolist(),[2024.,2024.])
        self.assertEqual(view['row_label'].tolist(),['Revenue','Interest'])
        q,a=compile_plan({'expression':"df['value'].sum()"},{'df':d},'Tính tổng')
        self.assertEqual(a,240.)
        self.assertEqual(evaluate_expression(q,{'df':d}),240.)
    def test_joined_ocr_amounts_and_decimal_percentage(self):
        d=pd.DataFrame({'value':[0.]*4,'raw_value':['12.345 67.890','(12.345)(67.890)','1.234%','1 234 567'],'source_scale':[1.]*4})
        self.assertEqual(normalize_view('df',d)['value'].tolist(),[12345.,-12345.,1.234,1234567.])

if __name__=='__main__':unittest.main()
