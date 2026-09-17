from scripts import _20_prepare_private_code_targets as targets


def _retrieval(*ids):
    return {
        qid: {
            "route": {
                "output_type": "number",
                "plan": {"op": "lookup"},
                "evidence_requirements": [],
            }
        }
        for qid in ids
    }


def test_target_cohort_keeps_unverified_single_vote_and_failed_rows():
    checkpoint = {
        1: {"status": "ok", "source": "llm_select", "votes": 1, "n_ok": 1},
        2: {"status": "failed", "source": "none", "votes": 0, "n_ok": 0},
        3: {"status": "ok", "source": "rule", "votes": 0, "n_ok": 0},
    }
    audit = {
        1: {"accepted": True}, 2: {"accepted": False}, 3: {"accepted": False},
    }

    ids, stats = targets.build_target_cohort(
        checkpoint, _retrieval(1, 2, 3), [audit])

    assert ids == [2]
    assert stats["weak_or_failed"] == 2
    assert stats["accepted_by_prior_verifiers"] == 1
