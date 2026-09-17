"""Temporal direction is distinct from an unordered difference magnitude."""
from pathlib import Path
import unittest

from road2ai_vifinqa.corpus import Corpus, load_questions
from road2ai_vifinqa.panel import FinancialPanel
from road2ai_vifinqa.paths import INDEX_PATH
from road2ai_vifinqa.template_solver import TemplateSolver, _temporal_delta_order


class TemporalDeltaTest(unittest.TestCase):
    def test_new_minus_old_in_both_textual_orders(self):
        self.assertEqual(_temporal_delta_order("Biến động số dư", (2021, 2020)), (0, 1))
        self.assertEqual(_temporal_delta_order("Thay đổi số dư", (2019, 2023)), (1, 0))

    def test_does_not_reinterpret_absolute_or_ambiguous_differences(self):
        self.assertIsNone(_temporal_delta_order("Độ lớn tuyệt đối của thay đổi", (2021, 2020)))
        self.assertIsNone(_temporal_delta_order("Chênh lệch giữa hai năm", (2021, 2020)))
        self.assertIsNone(_temporal_delta_order("Biến động", (2021, 2021)))
        self.assertIsNone(_temporal_delta_order("Biến động", (2021, 2022, 2023)))

    @unittest.skipUnless(Path(INDEX_PATH).exists(), "requires corpus")
    def test_option_changes_direction_but_not_source_values(self):
        questions = {q["id"]: q["question"] for q in load_questions()}
        with Corpus() as corpus:
            panel = FinancialPanel()
            legacy = TemplateSolver(corpus, panel)
            signed = TemplateSolver(corpus, panel, signed_temporal_changes=True)
            for qid in (624, 642, 649):
                with self.subTest(qid=qid):
                    old = legacy.solve(questions[qid], question_id=qid)
                    new = signed.solve(questions[qid], question_id=qid)
                    self.assertLess(new.answer, 0)
                    self.assertAlmostEqual(new.answer, -old.answer)
                    old_sources = {(s.doc_id, s.table_id, s.row_idx, s.col_idx, s.value) for s in old.sources}
                    new_sources = {(s.doc_id, s.table_id, s.row_idx, s.col_idx, s.value) for s in new.sources}
                    self.assertEqual(new_sources, old_sources)
            # A plain difference-magnitude question is deliberately unchanged.
            old = legacy.solve(questions[584], question_id=584)
            new = signed.solve(questions[584], question_id=584)
            self.assertEqual(old.answer, new.answer)


if __name__ == "__main__":
    unittest.main()
