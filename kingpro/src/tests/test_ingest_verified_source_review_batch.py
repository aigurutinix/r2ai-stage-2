import pytest

from scripts.ingest_verified_source_review_batch import normalize


def test_normalize_accepts_high_confidence_keep(tmp_path):
    artifact = tmp_path / "report.json"
    item = {
        "id": 1,
        "verdict": "keep",
        "current_answer": 12.34,
        "proposed_answer": 12.34,
        "confidence": 0.99,
        "proof": "Exact physical row and independent arithmetic agree.",
        "source_refs": ["DOC|10"],
    }
    result = normalize(item, artifact)
    assert result["answer"] == 12.34
    assert result["confidence"] == "high"


def test_normalize_rejects_keep_that_changes_answer(tmp_path):
    item = {
        "id": 1,
        "verdict": "keep",
        "current_answer": 12.34,
        "proposed_answer": 12.35,
        "confidence": 0.99,
        "proof": "Exact source proof is long enough.",
        "source_refs": ["DOC|10"],
    }
    with pytest.raises(ValueError, match="changes the answer"):
        normalize(item, tmp_path / "report.json")


def test_normalize_rejects_low_confidence(tmp_path):
    with pytest.raises(ValueError, match="confidence"):
        normalize(
            {
                "id": 1,
                "verdict": "keep",
                "current_answer": 1,
                "proposed_answer": 1,
                "confidence": 0.8,
                "proof": "Exact source proof is long enough.",
                "source_refs": ["DOC|10"],
            },
            tmp_path / "report.json",
        )


def test_normalize_accepts_cleanup_without_answer_change(tmp_path):
    result = normalize(
        {
            "id": 2,
            "verdict": "cleanup",
            "current_answer": 4.2,
            "proposed_answer": 4.2,
            "confidence": "high",
            "confidence_score": 0.98,
            "independent_recompute": "Physical source proves the answer while code needs cleanup.",
            "source_refs": ["DOC|20"],
        },
        tmp_path / "report.json",
    )
    assert result["mutation"] == "pending_batch_cleanup"
    assert "pending_batch_cleanup" in result["status"]
