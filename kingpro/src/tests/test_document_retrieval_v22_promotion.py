from __future__ import annotations

from scripts.audit_document_retrieval_v22_promotion import audit


def test_v22_bounded_retrieval_rules_pass_promotion_gate() -> None:
    report = audit()
    assert report["passed"], report
    assert all(report["checks"].values())
    assert report["metrics"]["macro_precision_after"] >= 0.975
    assert report["metrics"]["macro_recall_delta"] >= 0.002
    assert report["metrics"]["macro_f2_delta"] >= 0.0015
    assert report["metrics"]["missed_questions_after"] == 4
    assert report["metrics"]["multi_entity_raw_recall"] >= 0.997
