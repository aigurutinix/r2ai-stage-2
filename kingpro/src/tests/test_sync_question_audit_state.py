from scripts.sync_question_audit_state import compute_stats


def test_compute_stats_uses_ledger_as_authority() -> None:
    verdicts = {
        "2": {"mutation": "none"},
        "1": {"mutation": "pending_batch_retrieval_cleanup"},
    }
    public = {"queue": [{"id": 1}, {"id": 3}]}
    batch = {"queue": [{"id": 2}, {"id": 4}]}

    stats = compute_stats(verdicts, public, batch)

    assert stats["total_closed"] == 2
    assert stats["closed_no_change"] == 1
    assert stats["closed_pending_batch_cleanup"] == 1
    assert stats["next_question"] == 3
    assert stats["next_batch_question"] == 4
    assert stats["latest_closed"] == [2, 1]
