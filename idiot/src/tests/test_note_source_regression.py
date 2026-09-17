"""Source-based regression tests, not hidden competition accuracy labels."""
from dataclasses import asdict
from pathlib import Path
import unittest

import pandas as pd
from road2ai_vifinqa.corpus import Corpus, load_questions
from road2ai_vifinqa.hard_note_solver import solve_note
from road2ai_vifinqa.paths import INDEX_PATH
from road2ai_vifinqa.panel import FinancialPanel
from road2ai_vifinqa.submission import evaluate_expression
from road2ai_vifinqa.template_solver import TemplateSolver, _ac


@unittest.skipUnless(Path(INDEX_PATH).exists(), "requires the local official corpus index")
class NoteSourceRegressionTest(unittest.TestCase):
    def test_vgt_selector_uses_related_party_subset_in_every_year(self):
        question = next(q["question"] for q in load_questions() if q["id"] == 495)
        with Corpus() as corpus:
            result = solve_note(question, 495, corpus)
        sources = [asdict(s) for s in result.sources]
        selectors = [s for s in sources if s["retrieval_phrase"] == "related-party other short-term receivables total"]
        source_2021 = next(s for s in selectors if s["report_year"] == 2021)
        self.assertEqual((source_2021["table_id"], source_2021["row_idx"], source_2021["col_idx"]), (26, 10, 1))
        self.assertEqual(source_2021["vnd_value"], 248590459577)
        self.assertEqual(max(selectors, key=lambda s: s["vnd_value"])["report_year"], 2020)
        target = next(s for s in sources if s["report_year"] == 2020 and s["retrieval_phrase"] == "minimum operating-lease payments total")
        self.assertAlmostEqual(result.answer, target["vnd_value"] / 1e9)
        self.assertAlmostEqual(evaluate_expression(result.pandas_query, {"df": pd.DataFrame(sources)}), result.answer)
        changed = pd.DataFrame(sources)
        changed.loc[changed.candidate_id == "c0003", "vnd_value"] = 1e12
        target_2021 = next(s for s in sources if s["report_year"] == 2021 and s["retrieval_phrase"] == "minimum operating-lease payments total")
        self.assertAlmostEqual(evaluate_expression(result.pandas_query, {"df": changed}), target_2021["vnd_value"] / 1e9)

    def test_qns_tie_break_uses_gross_principal_not_allowance(self):
        question = next(q["question"] for q in load_questions() if q["id"] == 501)
        with Corpus() as corpus:
            result = solve_note(question, 501, corpus)
        source = next(s for s in result.sources if s.report_year == 2021 and s.retrieval_phrase == "gross overdue receivables total tie-break")
        self.assertEqual((source.table_id, source.row_idx, source.col_idx), (21, 10, 1))
        self.assertEqual(source.vnd_value, 19214598721)
        self.assertEqual(result.answer, 2023)
        frame = pd.DataFrame(asdict(s) for s in result.sources)
        self.assertEqual(evaluate_expression(result.pandas_query, {"df": frame}), 2023)
        frame.loc[frame.candidate_id == "c0006", "vnd_value"] = 1e12
        self.assertEqual(evaluate_expression(result.pandas_query, {"df": frame}), 2021)

    def test_corrected_note_sources_and_expressions(self):
        questions = {q["id"]: q["question"] for q in load_questions()}
        with Corpus() as corpus:
            results = {q: solve_note(questions[q], q, corpus) for q in (502, 506, 521, 526)}
        for qid, result in results.items():
            with self.subTest(qid=qid):
                frame = pd.DataFrame(asdict(s) for s in result.sources)
                self.assertAlmostEqual(evaluate_expression(result.pandas_query, {"df": frame}), result.answer)
        bank = next(s for s in results[502].sources if s.report_year == 2021)
        self.assertEqual((bank.table_id, bank.row_idx, bank.col_idx, bank.vnd_value), (11, 3, 1, 16633673))
        tangible = [s for s in results[506].sources if "selector" in s.retrieval_phrase]
        self.assertEqual(len(tangible), 5)
        self.assertTrue(all("tangible" in s.retrieval_phrase for s in tangible))
        donor = next(s for s in results[521].sources if s.report_year == 2024)
        self.assertEqual((donor.table_id, donor.row_idx, donor.col_idx, donor.vnd_value), (3, 2, 4, 245549342427))
        self.assertAlmostEqual(results[521].answer, 299.780784516)
        eps = next(s for s in results[526].sources if s.report_year == 2023)
        self.assertEqual((eps.table_id, eps.row_idx, eps.col_idx, eps.raw_number), (45, 3, 1, 115))

    def test_explicit_comparatives_keep_period_separate_from_document(self):
        with Corpus() as corpus:
            solver = TemplateSolver(corpus, FinancialPanel())
            donor = "HDG_financial_statements_2024_consolidated"
            cell = solver._load_audited_cell(_ac("HDG", 2023, donor, 7, 6, 4, prior_year_comparative=True))
            self.assertIsNotNone(cell)
            self.assertEqual((cell.year, cell.doc_id, cell.value), (2023, donor, 40301874302))
            self.assertIsNone(solver._load_audited_cell(_ac("HDG", 2023, donor, 7, 6, 4)))
            self.assertIsNone(solver._load_audited_cell(_ac("HDG", 2023, donor, 7, 6, 3, prior_year_comparative=True)))
            self.assertIsNone(solver._load_audited_cell(_ac("HDG", 2022, donor, 7, 6, 4, prior_year_comparative=True)))
            self.assertIsNone(solver._load_audited_cell(_ac("VGT", 2023, donor, 7, 6, 4, prior_year_comparative=True)))
            question = next(q["question"] for q in load_questions() if q["id"] == 904)
            result = solver.solve(question, question_id=904)
        self.assertEqual(result.answer, 2024)
        self.assertEqual([s.year for s in result.sources], [2020, 2023, 2024])


if __name__ == "__main__":
    unittest.main()
