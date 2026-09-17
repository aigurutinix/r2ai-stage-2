import unittest
import pandas as pd
from road2ai_vifinqa.direct_evidence_query import direct_plan,temporal_pair_plan,direct_percentage_plan
from road2ai_vifinqa.compliant_query import compile_plan

class DirectSourceTest(unittest.TestCase):
    def setUp(self):self.frames={'df':pd.DataFrame({'raw_number':[1234.], 'source_scale':[1e6], 'answer_value':[9999]})}
    def test_reads_and_converts_the_source(self):
        question='Doanh thu của công ty ABC năm 2024 là bao nhiêu tỷ đồng?'
        plan=direct_plan(question,self.frames)
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(compile_plan(plan,self.frames,question)[1],1.234)
    def test_does_not_handle_comparison_or_ratio(self):
        for q in ['Doanh thu tăng bao nhiêu tỷ đồng?', 'A và B có tổng bao nhiêu tỷ đồng?', 'Tỷ lệ lợi nhuận bao nhiêu %?']:
            self.assertIsNone(direct_plan(q,self.frames))
    def test_requires_single_raw_cell(self):
        self.assertIsNone(direct_plan('Doanh thu bao nhiêu tỷ đồng?',{'df':pd.concat([self.frames['df']]*2)}))
    def test_raw_text_source_is_reconstructed_not_prepared_value(self):
        frames={'df':pd.DataFrame({'raw_value':['1.234.000'], 'source_scale':[1e3], 'value':[123.]})}
        question='Doanh thu là bao nhiêu tỷ đồng?'
        plan=direct_plan(question,frames)
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(compile_plan(plan,frames,question)[1],1.234)
    def test_temporal_direction_and_percent(self):
        frames={'df':pd.DataFrame({'value':[100e9,120e9], 'source_scale':[1,1], 'label':['Revenue']*2,'year':[2020,2021]})}
        q='Doanh thu năm 2021 tăng bao nhiêu phần trăm so với năm 2020?'
        p=temporal_pair_plan(q,frames)
        self.assertAlmostEqual(compile_plan(p,frames,q)[1],20.)
        q='Doanh thu năm 2020 thấp hơn năm 2021 bao nhiêu tỷ đồng?'
        self.assertAlmostEqual(compile_plan(temporal_pair_plan(q,frames),frames,q)[1],20.)
    def test_ambiguous_difference_not_guessed(self):
        frames={'df':pd.DataFrame({'value':[100e9,120e9], 'source_scale':[1,1], 'label':['Revenue']*2,'year':[2020,2021]})}
        self.assertIsNone(temporal_pair_plan('Chênh lệch năm 2020 và 2021 bao nhiêu tỷ đồng?',frames))
    def test_ordered_difference_ignores_footnote_marker(self):
        frames={'df':pd.DataFrame({'value':[120e9,100e9],'source_scale':[1,1],
            'label':['Chi phí lãi vay (*)','Chi phí lãi vay'],'year':[2021,2020]})}
        question='Hiệu số giữa chi phí lãi vay năm 2021 và năm 2020 là bao nhiêu tỷ đồng?'
        self.assertEqual(compile_plan(temporal_pair_plan(question,frames),frames,question)[1],20.)
    def test_printed_percentage_is_read_not_hardcoded(self):
        frames={'df':pd.DataFrame({'raw_value':['45%'],'raw_number':[45.]})}
        q='Tỷ lệ sở hữu của ABC là bao nhiêu %?'
        p=direct_percentage_plan(q,frames)
        self.assertEqual(compile_plan(p,frames,q)[1],45.)
        self.assertIsNone(direct_percentage_plan('Tỷ lệ sở hữu tăng bao nhiêu %?',frames))

if __name__=='__main__':unittest.main()
