from __future__ import annotations

import math
import unittest

from road2ai_vifinqa.easy_reranker import (
    EASY_RERANKER_MODEL,
    EASY_RERANKER_VALIDATION,
    MODEL_RESOURCE,
    generator_feature_vectors,
    score_easy_candidates,
)
from road2ai_vifinqa.easy_solver import (
    EASY_AUDITED_OVERRIDES,
    EasyCandidate,
    shortlist_easy_candidates,
)


def _candidate(
    candidate_id: str,
    *,
    table_id: int,
    row_idx: int,
    col_idx: int,
    row_label: str,
    column_header: str,
    retrieval_score: float,
) -> EasyCandidate:
    return EasyCandidate(
        candidate_id=candidate_id,
        ticker="ABC",
        report_year=2019,
        scope="consolidated",
        doc_id="ABC_financial_statements_2019_consolidated",
        table_id=table_id,
        table_rows=12,
        row_idx=row_idx,
        col_idx=col_idx,
        row_label=row_label,
        section="2019 triệu đồng | 2018 triệu đồng",
        column_header=column_header,
        table_context="Báo cáo kết quả hoạt động kinh doanh hợp nhất",
        raw_value="1.234",
        raw_number=1234.0,
        source_scale=1_000_000.0,
        requested_scale=1_000_000.0,
        answer_value=1234.0,
        retrieval_score=retrieval_score,
    )


class EasyRerankerDeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.question = "Tổng chi phí hoạt động của ABC năm 2019 là bao nhiêu triệu đồng?"
        self.candidates = [
            _candidate(
                "e000001",
                table_id=3,
                row_idx=4,
                col_idx=2,
                row_label="TỔNG CHI PHÍ HOẠT ĐỘNG",
                column_header="2019 triệu đồng",
                retrieval_score=19.0,
            ),
            _candidate(
                "e000002",
                table_id=3,
                row_idx=4,
                col_idx=3,
                row_label="TỔNG CHI PHÍ HOẠT ĐỘNG",
                column_header="2018 triệu đồng",
                retrieval_score=18.0,
            ),
            _candidate(
                "e000003",
                table_id=9,
                row_idx=2,
                col_idx=1,
                row_label="Chi phí khác",
                column_header="2019 triệu đồng",
                retrieval_score=22.0,
            ),
            _candidate(
                "e000004",
                table_id=12,
                row_idx=7,
                col_idx=1,
                row_label="Tổng cộng",
                column_header="2019 triệu đồng",
                retrieval_score=14.0,
            ),
        ]
        self.bm25 = {
            (self.candidates[0].doc_id, 3, 4): 12.5,
            (self.candidates[0].doc_id, 9, 2): 4.0,
            (self.candidates[0].doc_id, 12, 7): 7.0,
        }

    def test_scores_and_shortlist_are_bitwise_deterministic(self) -> None:
        first_scores = score_easy_candidates(self.question, self.candidates, self.bm25)
        second_scores = score_easy_candidates(self.question, self.candidates, self.bm25)
        self.assertEqual(first_scores, second_scores)
        self.assertEqual(
            first_scores,
            {
                "e000001": 8.377388427516848,
                "e000002": 7.989863218243307,
                "e000003": 6.693172579969305,
                "e000004": 5.495556583215291,
            },
        )
        self.assertTrue(all(math.isfinite(value) for value in first_scores.values()))

        first = shortlist_easy_candidates(
            self.candidates,
            question=self.question,
            bm25_row_scores=self.bm25,
            use_learned_reranker=True,
            max_rows=2,
            max_candidates=3,
        )
        second = shortlist_easy_candidates(
            self.candidates,
            question=self.question,
            bm25_row_scores=self.bm25,
            use_learned_reranker=True,
            max_rows=2,
            max_candidates=3,
        )
        self.assertEqual(
            [(item.candidate_id, item.retrieval_score) for item in first],
            [(item.candidate_id, item.retrieval_score) for item in second],
        )
        self.assertEqual([item.candidate_id for item in first], ["e000001", "e000002", "e000003"])

    def test_v2_replaces_v1_instead_of_stacking_bm25_weights(self) -> None:
        vectors = generator_feature_vectors(self.question, self.candidates)
        self.assertEqual(len(vectors), len(self.candidates))
        self.assertTrue(all(len(vector) == 58 for vector in vectors))
        self.assertTrue(all(math.isfinite(value) for vector in vectors for value in vector))
        scores = score_easy_candidates(self.question, self.candidates, self.bm25)
        perturbed = score_easy_candidates(
            self.question,
            self.candidates,
            {key: value + 1_000_000.0 for key, value in self.bm25.items()},
        )
        self.assertEqual(scores, perturbed)

    def test_learned_mode_requires_question_and_bm25(self) -> None:
        with self.assertRaises(ValueError):
            shortlist_easy_candidates(self.candidates, use_learned_reranker=True)


class EasyRerankerValidationProvenanceTest(unittest.TestCase):
    def test_reported_metrics_are_explicitly_out_of_fold(self) -> None:
        validation = EASY_RERANKER_VALIDATION
        self.assertIsInstance(validation, dict)
        self.assertEqual(
            validation["protocol"],
            "five_fold_question_disjoint_cross_validation_qid_mod_5",
        )
        self.assertEqual(
            validation["reported_predictions"],
            "out_of_fold_from_fold_specific_models",
        )
        self.assertFalse(validation["reported_metrics_are_in_sample"])
        self.assertFalse(validation["full_fit_weights_used_for_reported_metrics"])
        self.assertEqual(validation["legacy_baseline_production_style_shortlist_hits"], 72)
        self.assertEqual(validation["v1_checked_in_reported_shortlist_hits"], 91)
        self.assertEqual(validation["v2_production_style_shortlist_hits"], 95)
        self.assertEqual(validation["n_questions"], 101)
        self.assertEqual(
            validation["shortlist_policy"],
            "rank_complete_rows_max_28_then_truncate_to_64_cells",
        )
        self.assertEqual(
            validation["secondary_protocol"],
            "five_fold_issuer_grouped_cross_validation",
        )

    def test_declared_validation_folds_are_question_disjoint(self) -> None:
        ids = {int(value) for value in EASY_RERANKER_MODEL["training_question_ids"]}
        self.assertEqual(len(ids), 101)
        for fold in range(5):
            held_out = {qid for qid in ids if qid % 5 == fold}
            training = ids - held_out
            self.assertTrue(held_out)
            self.assertTrue(training)
            self.assertTrue(held_out.isdisjoint(training))
            self.assertEqual(held_out | training, ids)

    def test_review_corrections_are_not_reranker_training_data(self) -> None:
        training_ids = {
            int(value) for value in EASY_RERANKER_MODEL["training_question_ids"]
        }
        self.assertTrue({66, 95, 115}.isdisjoint(training_ids))
        self.assertNotIn(66, EASY_AUDITED_OVERRIDES)
        self.assertEqual(
            EASY_AUDITED_OVERRIDES[80][1],
            (("NVL_financial_statements_2020_consolidated", 45, 4, 1),),
        )
        self.assertEqual(
            EASY_AUDITED_OVERRIDES[95][1],
            (("HDG_financial_statements_2015_consolidated", 52, 2, 1),),
        )
        self.assertEqual(
            EASY_AUDITED_OVERRIDES[115][1],
            (("CTG_financial_statements_2019_separate", 7, 14, 2),),
        )

    def test_feature_and_weight_schema_is_fixed(self) -> None:
        names = EASY_RERANKER_MODEL["feature_names"]
        weights = EASY_RERANKER_MODEL["effective_raw_feature_weights"]
        self.assertEqual(MODEL_RESOURCE, "easy_reranker_v2.json")
        self.assertEqual(EASY_RERANKER_MODEL["schema"], 2)
        self.assertEqual(len(names), 58)
        self.assertEqual(len(names), len(weights))
        self.assertEqual(len(set(names)), len(names))


if __name__ == "__main__":
    unittest.main()
