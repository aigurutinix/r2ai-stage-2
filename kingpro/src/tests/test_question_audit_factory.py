from pathlib import Path

from scripts.run_question_audit_factory import (
    build_clusters,
    arithmetic_invariant_flags,
    intent_source_flags,
    normalized_ast_hash,
    positional_fallback,
    priority,
    question_fingerprint,
    source_bundle_hash,
)
from scripts.run_audit_factory_pipeline import select_deep_ids


def test_gross_question_net_source_is_high_priority() -> None:
    flags = intent_source_flags(
        "Doanh thu bán hàng và cung cấp dịch vụ năm 2023 là bao nhiêu?",
        ["kqkd:10"],
    )

    assert flags == [
        {
            "family": "gross_question_net_source",
            "severity": "answer_change_possible",
            "question_intent": "kqkd:01",
            "declared_source": "kqkd:10",
        }
    ]


def test_dead_qid_does_not_change_normalized_ast() -> None:
    left = "qid = 485\nx = 1\nresult = float(x)"
    right = "qid = 561\nx = 1\nresult = float(x)"

    assert normalized_ast_hash(left) == normalized_ast_hash(right)


def test_fingerprint_changes_with_evidence_content_hash() -> None:
    row = {"id": 1, "answer": 1.0, "pandas_query": "result=1.0"}
    first = [{"variable": "df1", "csv_path": "data/a.csv", "exists": True, "sha256": "A"}]
    second = [{"variable": "df1", "csv_path": "data/a.csv", "exists": True, "sha256": "B"}]

    assert question_fingerprint(row, first, "TOOL") != question_fingerprint(row, second, "TOOL")


def test_source_bundle_cluster_is_candidate_only() -> None:
    hashes = [{"sha256": "ABC"}]
    bundle = source_bundle_hash(hashes)
    records = [
        {
            "id": 485,
            "answer": 0.81,
            "source_bundle_hash": bundle,
            "normalized_ast_hash": "AST",
            "relevant_tables": ["T|1"],
        },
        {
            "id": 561,
            "answer": 0.81,
            "source_bundle_hash": bundle,
            "normalized_ast_hash": "AST",
            "relevant_tables": ["T|1"],
        },
    ]

    clusters = build_clusters(records)

    assert clusters[0]["question_ids"] == [485, 561]
    assert clusters[0]["normalized_ast_equal"] is True
    assert "requires review" in clusters[0]["claim_limit"]


def test_positional_fallback_resolves_legacy_iloc(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "table.csv").write_text(
        "0,1\nChi phí thuế TNDN hiện hành,43.725.665.762\n",
        encoding="utf-8",
    )
    row = {
        "id": 95,
        "question": "Chi phí thuế hiện hành là bao nhiêu?",
        "evidence": [{"variable": "df1", "csv_path": "data/table.csv"}],
        "pandas_query": "result = df1.iloc[0, 1]",
    }

    result = positional_fallback(row, tmp_path)

    assert result["status"] == "positional_pass"
    assert result["reads"][0]["label"] == "Chi phí thuế TNDN hiện hành"
    assert result["reads"][0]["raw"] == "43.725.665.762"


def test_priority_puts_intent_source_mismatch_first() -> None:
    score, reasons = priority(
        {
            "intent_source_flags": [{"family": "gross_question_net_source"}],
            "runtime": {"string": {"status": "pass"}},
            "physical_audit": {"status": "pass"},
            "unused_evidence": None,
        }
    )

    assert score == 100
    assert reasons == ["intent_source_mismatch"]


def test_named_counterparty_generic_row_is_flagged_but_exact_row_is_not() -> None:
    question = (
        "Vay dài hạn với Công ty Cổ phần Hoàng Anh Gia Lai của công ty mẹ HNG "
        "cuối năm 2017 là bao nhiêu?"
    )
    risky = intent_source_flags(
        question,
        ["note:hag_long_term_loan_ending"],
        [{"label": "Vay dài hạn"}],
    )
    exact = intent_source_flags(
        "Số dư phải thu từ Công ty Cổ phần Bao bì Dầu khí Việt Nam của DCM là bao nhiêu?",
        ["note:customer_receivable"],
        [{"label": "Công ty Cổ phần Bao bì Dầu khí Việt Nam"}],
    )

    assert any(item["family"] == "named_entity_context_missing" for item in risky)
    assert not any(item["family"] == "named_entity_context_missing" for item in exact)


def test_named_counterparty_can_bind_from_group_ancestor() -> None:
    flags = intent_source_flags(
        "Giá trị mua hàng hóa và dịch vụ từ Công ty TNHH Coats Phong Phú là bao nhiêu?",
        ["note:coats_phong_phu_purchases"],
        [
            {
                "label": "Mua hàng hóa và dịch vụ",
                "entity_ancestor": "Công ty TNHH Coats Phong Phú",
            }
        ],
    )

    assert not any(item["family"] == "named_entity_context_missing" for item in flags)


def test_pipeline_prioritizes_factory_score_then_preserves_queue_order() -> None:
    priority = [{"id": 3}, {"id": 1}]

    assert select_deep_ids(priority, [1, 2, 3, 4], 3) == [3, 1, 2]


def test_component_over_total_invariant_flags_q826_shape() -> None:
    flags = arithmetic_invariant_flags(
        "Tỷ trọng giá vốn cho thuê trên tổng giá vốn cao nhất là năm nào?",
        "result = max((abs(v0 / v4), 2016), (abs(v1 / v5), 2019))[1]",
        {"v0": 1632.119, "v4": 865.066, "v1": 100.0, "v5": 200.0},
    )

    assert flags[0]["family"] == "component_exceeds_total"
    assert flags[0]["ratio"] > 1.8
