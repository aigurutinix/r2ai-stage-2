import json

import scripts.solve_leaderboard_answer_constraints as solver
from scripts.solve_leaderboard_answer_constraints import Observation, answer_key, solve_constraints


def observation(sid, count, q1, q2):
    return Observation(
        submission_id=sid,
        candidate=f"c{sid}",
        candidate_sha256=str(sid),
        answer_count=count,
        answers={1: answer_key(q1), 2: answer_key(q2)},
        score_source="test",
        match_kind="exact",
    )


def test_solver_forces_public_answer_variants_from_aggregate_counts():
    # Hidden truth is q1=11 and q2=20.  The solver sees only aggregate counts.
    observations = [
        observation(1, 1, 10, 20),
        observation(2, 2, 11, 20),
        observation(3, 1, 11, 21),
    ]
    reference = {1: answer_key(10), 2: answer_key(20)}

    result = solve_constraints(observations, reference)

    assert result["forced_reference_repairs"] == [
        {"id": 1, "reference_answer": "num:10.00", "forced_answer": "num:11.00"}
    ]


def test_answer_key_clusters_signed_zero_and_two_decimal_values():
    assert answer_key(-0.0) == answer_key(0.0) == "num:0.00"
    assert answer_key(1.234) == "num:1.23"


def test_single_question_zero_delta_forces_both_variants_wrong():
    observations = [
        observation(1, 1, 10, 20),
        observation(2, 1, 11, 20),
    ]
    result = solve_constraints(observations, {1: answer_key(10), 2: answer_key(20)})
    q1 = [item for item in result["forced_variants"] if item["id"] == 1]
    assert len(q1) == 2
    assert all(item["forced_correct"] is False for item in q1)


def test_running_state_does_not_clobber_finished_api_score(tmp_path, monkeypatch):
    score_dir = tmp_path / "build"
    score_dir.mkdir()
    (score_dir / "leaderboard_scores_finished.json").write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "id": 3742,
                        "filename": "sub_v276_q638_fix.zip",
                        "scores": {"ANSWER_ACCURACY": 0.7115},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    state_dir = tmp_path / "knowledge" / "vothuong"
    state_dir.mkdir(parents=True)
    (state_dir / "current_state.json").write_text(
        json.dumps(
            {
                "authoritative_public_state": {
                    "latest_submission": {
                        "version": "v276",
                        "submission_id": 3742,
                        "filename": "sub_v276_q638_fix.zip",
                        "scores": {"answer_accuracy": None},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(solver, "ROOT", tmp_path)
    rows, _ = solver.collect_score_rows(score_dir)
    assert len(rows) == 1
    assert rows[0]["id"] == 3742
    assert rows[0]["scores"]["ANSWER_ACCURACY"] == 0.7115
