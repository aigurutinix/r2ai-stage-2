from __future__ import annotations

from scripts.audit_document_retrieval_v21_promotion import audit


def test_v21_bounded_comparative_series_rule_passes_promotion_gate() -> None:
    report = audit()
    assert report["passed"], report
    assert all(report["checks"].values())
    assert report["metrics"]["macro_recall_delta"] > 0
    assert report["metrics"]["macro_f2_delta"] > 0
    assert report["metrics"]["missed_questions_after"] == 9
    assert report["metrics"]["multi_entity_effective_grounded_recall"] >= 0.99
