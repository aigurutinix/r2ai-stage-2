from __future__ import annotations

from scripts.audit_private_proxy_cohorts import _archetype, audit


def test_private_proxy_archetypes_are_semantic_not_id_based() -> None:
    assert _archetype("Doanh thu thuần của VNM năm 2024 là bao nhiêu?") == "direct_lookup"
    assert _archetype("Tỷ lệ CFO trên doanh thu của VNM là bao nhiêu phần trăm?") == "ratio"
    assert _archetype("Công ty có doanh thu cao nhất trong nhóm là công ty nào?") == "conditional_extreme"


def test_private_proxy_cohort_gate_passes_after_v22_bounded_retrieval() -> None:
    report = audit()
    assert report["passed"], report
    assert report["global"]["questions"] == 1012
    assert report["global"]["compiler_accepted"] >= 106
    assert report["global"]["compiler_precision"] == 1.0
    assert report["global"]["effective_grounded_macro_recall"] >= report["global"]["document_macro_recall"]
    assert report["risk_flags"] == []
    assert report["retrieval_observations"] == []
    multi = report["cohorts"]["entity_folds"]["multi_entity_or_unresolved"]
    assert multi["document_macro_recall"] == 0.997701
    assert multi["effective_grounded_macro_recall"] == 0.997701
    assert all("leaderboard" not in key for key in report["input_hashes"])
