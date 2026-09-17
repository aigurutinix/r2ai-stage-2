from pathlib import Path
import unittest

from road2ai_vifinqa.corpus import Corpus, load_questions
from road2ai_vifinqa.panel import FinancialPanel
from road2ai_vifinqa.paths import INDEX_PATH
from road2ai_vifinqa.template_solver import TemplateSolver


@unittest.skipUnless(Path(INDEX_PATH).exists(), "requires source corpus")
class LiteralRatioContractTest(unittest.TestCase):
    def test_unitless_and_gross_revenue_contracts(self):
        questions = {q["id"]: q["question"] for q in load_questions()}
        with Corpus() as corpus:
            old = TemplateSolver(corpus, FinancialPanel(), signed_temporal_changes=True)
            new = TemplateSolver(corpus, old.panel, signed_temporal_changes=True, literal_ratio_contracts=True)
            before = old.solve(questions[672], question_id=672)
            after = new.solve(questions[672], question_id=672)
            self.assertAlmostEqual(after.answer, before.answer / 100)
            self.assertEqual(after.sources, before.sources)
            # Adding an explicit percent request restores percent output.
            explicit = new.solve(questions[672] + " Tính theo phần trăm.", question_id=672)
            self.assertAlmostEqual(explicit.answer, before.answer)
            share = new.solve(questions[711], question_id=711)
            self.assertEqual((share.sources[1].table_id, share.sources[1].row_idx), (41, 7))
            self.assertAlmostEqual(share.answer, share.sources[0].value / share.sources[1].value * 100)
            # A net-revenue question must not switch to gross sales.
            net_question = questions[711].replace("doanh thu bán hàng", "doanh thu thuần bán hàng")
            net = new.solve(net_question, question_id=711)
            self.assertEqual((net.sources[1].table_id, net.sources[1].row_idx), (7, 3))
            for qid in (661, 678, 706):
                self.assertEqual(old.solve(questions[qid], question_id=qid).answer,
                                 new.solve(questions[qid], question_id=qid).answer)


if __name__ == "__main__":
    unittest.main()
