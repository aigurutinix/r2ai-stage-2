from scripts.collect_compact_audit_results import validate_result


def test_validate_compact_result_contract() -> None:
    value = {
        "id": 508,
        "verdict": "keep",
        "confidence": "high",
        "one_line_reason": "source exact",
        "source_refs": ["A|1"],
        "cluster_ids": [],
    }

    assert validate_result(value, 508) == []


def test_rejects_verbose_or_wrong_id_result() -> None:
    value = {
        "id": 1,
        "verdict": "keep",
        "confidence": "high",
        "one_line_reason": "x" * 241,
        "source_refs": [],
        "cluster_ids": [],
    }

    assert validate_result(value, 2) == ["id_mismatch", "invalid_reason"]
