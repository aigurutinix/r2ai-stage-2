from pathlib import Path
import unittest

from road2ai_vifinqa.corpus import Corpus, load_questions
from road2ai_vifinqa.panel import FinancialPanel
from road2ai_vifinqa.paths import INDEX_PATH
from road2ai_vifinqa.template_solver import TemplateSolver, _ordered_comparison_indices


class OrderedComparisonTest(unittest.TestCase):
    def test_comparison_preserves_requested_order(self):
        points = (("AAA", 2024), ("BBB", 2024))
        self.assertEqual(_ordered_comparison_indices("AAA chênh lệch so với BBB", points, points), (0, 1))
        self.assertEqual(_ordered_comparison_indices("AAA chênh lệch so với BBB", points[::-1], points), (1, 0))
        self.assertEqual(_ordered_comparison_indices("So với AAA, BBB chênh lệch bao nhiêu", points, points), (1, 0))

    def test_magnitude_and_incomplete_mapping_are_not_reinterpreted(self):
        points = (("AAA", 2024), ("BBB", 2024))
        for text in ("Chênh lệch tuyệt đối so với BBB", "Độ lớn chênh lệch so với BBB",
                     "Chênh lệch nhau so với BBB", "Chênh lệch giữa AAA và BBB"):
            self.assertIsNone(_ordered_comparison_indices(text, points, points))
        self.assertIsNone(_ordered_comparison_indices("AAA chênh lệch so với BBB", points, points[:1]))

    def test_opt_in_temporal_contrast_has_ordered_years(self):
        points = (("AAA", 2025), ("AAA", 2024))
        text = "Chênh lệch vốn chủ sở hữu giữa năm 2025 và năm 2024"
        self.assertIsNone(_ordered_comparison_indices(text, points, points))
        self.assertEqual(_ordered_comparison_indices(text, points[::-1], points, temporal_contrasts=True), (1, 0))
        self.assertIsNone(_ordered_comparison_indices("Độ chênh lệch giữa hai năm", points, points, temporal_contrasts=True))

    @unittest.skipUnless(Path(INDEX_PATH).exists(), "requires corpus")
    def test_net_receivable_group_retains_allowance_deduction(self):
        questions = {q["id"]: q["question"] for q in load_questions()}
        with Corpus() as corpus:
            panel = FinancialPanel()
            old = TemplateSolver(corpus, panel, signed_temporal_changes=True, ordered_comparisons=True)
            new = TemplateSolver(corpus, panel, signed_temporal_changes=True, ordered_comparisons=True, temporal_contrasts=True)
            for qid in (584, 618, 623, 653):
                before = old.solve(questions[qid], question_id=qid)
                after = new.solve(questions[qid], question_id=qid)
                self.assertAlmostEqual(after.answer, -before.answer)
                self.assertEqual(set(after.sources), set(before.sources))
            self.assertEqual(old.solve(questions[592], question_id=592).answer,
                             new.solve(questions[592], question_id=592).answer)

    @unittest.skipUnless(Path(INDEX_PATH).exists(), "requires corpus")
    def test_sources_unchanged_and_negative_balances_untouched(self):
        questions = {q["id"]: q["question"] for q in load_questions()}
        with Corpus() as corpus:
            panel = FinancialPanel()
            legacy = TemplateSolver(corpus, panel, signed_temporal_changes=True)
            ordered = TemplateSolver(corpus, panel, signed_temporal_changes=True, ordered_comparisons=True)
            for qid in (742, 758, 781, 790, 793, 799):
                with self.subTest(qid=qid):
                    before = legacy.solve(questions[qid], question_id=qid)
                    after = ordered.solve(questions[qid], question_id=qid)
                    self.assertLess(after.answer, 0)
                    self.assertAlmostEqual(after.answer, -before.answer)
                    self.assertEqual(set(before.sources), set(after.sources))
            for qid in (584, 780, 802):
                before = legacy.solve(questions[qid], question_id=qid)
                after = ordered.solve(questions[qid], question_id=qid)
                self.assertEqual(before.answer, after.answer)


if __name__ == "__main__":
    unittest.main()
