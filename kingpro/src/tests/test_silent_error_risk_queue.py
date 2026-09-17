import json

from scripts.build_silent_error_risk_queue import artifact_compatibility, build_record


def base_row(**overrides):
    row = {
        "id": 7,
        "question": "Năm 2023 và 2024 công ty AAA tăng trưởng bao nhiêu?",
        "answer": 10.0,
        "relevant_docs": ["AAA_financial_statements_2024_consolidated"],
        "relevant_tables": ["AAA_financial_statements_2024_consolidated|10"],
        "pandas_query": "result = round((df.iloc[0, 1] / df.iloc[0, 2] - 1) * 100, 2)",
    }
    row.update(overrides)
    return row


def build(tmp_path, row=None, **payloads):
    return build_record(
        row or base_row(),
        tmp_path,
        payloads.get("fast", {}),
        payloads.get("deep", {}),
        payloads.get("reviewed", {}),
        payloads.get("missing", {}),
        payloads.get("cross_report", {}),
        payloads.get("rounding", {}),
        payloads.get("unit", {}),
        payloads.get("panel", {}),
        payloads.get("independent", {}),
    )


def test_missing_assurance_is_visible_not_treated_as_correct(tmp_path):
    result = build(tmp_path)
    assert result["state"] == "under_tested"
    assert result["challenge_matrix"]["physical_lineage"]["status"] == "blocked"
    assert result["challenge_matrix"]["independent_solver"]["status"] == "untested"
    assert result["residual_risk"] > 0


def test_independent_disagreement_raises_risk(tmp_path):
    baseline = build(tmp_path)
    disputed = build(
        tmp_path,
        independent={
            "all_disagreements": [
                {"id": 7, "dependency_groups": ["solver_a", "solver_b"]}
            ]
        },
    )
    assert disputed["residual_risk"] > baseline["residual_risk"]
    assert disputed["state"] == "challenged_needs_adjudication"


def test_physical_and_dual_runtime_reduce_residual_risk(tmp_path):
    compact = tmp_path / "data" / "q7_source_cells.csv"
    compact.parent.mkdir()
    compact.write_text("metric_key,raw\nx,1\n", encoding="utf-8")
    deep = {
        7: {
            "runtime": {"string": {"status": "pass"}, "official_typed": {"status": "pass"}},
            "physical_audit": {"status": "pass"},
            "intent_source_flags": [],
            "arithmetic_invariant_flags": [],
        }
    }
    result = build(tmp_path, deep=deep, reviewed={"7": {"status": "source_confirmed", "confidence": "high"}})
    assert result["assurance_score"] >= 49
    assert result["challenge_matrix"]["dual_runtime"]["status"] == "pass"
    assert result["challenge_matrix"]["physical_lineage"]["status"] == "pass"
    assert result["residual_risk"] < 20


def test_source_proven_answer_change_is_not_counted_as_assurance(tmp_path):
    result = build(
        tmp_path,
        reviewed={"7": {"status": "source_confirmed_answer_change", "answer": 11.0}},
    )
    assert result["state"] == "hard_signal_needs_source_proof"
    assert any(item["family"] == "durable_answer_change" for item in result["contradictions"])


def test_terminal_source_review_closes_old_disagreement_signal(tmp_path):
    result = build(
        tmp_path,
        independent={"all_disagreements": [{"id": 7, "dependency_groups": ["a", "b"]}]},
        reviewed={"7": {"status": "source_confirmed_no_change", "answer": 10.0, "mutation": "none"}},
    )
    assert result["state"] == "source_adjudicated"


def test_documented_source_gold_conflict_is_terminal_not_reopened(tmp_path):
    result = build(
        tmp_path,
        reviewed={"7": {"status": "source_confirmed_change_not_applied_public_gold_conflict", "answer": 11.0, "mutation": "not_applied_public_gold_conflict"}},
    )
    assert result["state"] == "source_adjudicated"
    assert not any(item["family"] == "durable_answer_change" for item in result["contradictions"])


def test_artifact_compatibility_is_exact_per_question(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    old_rows = [base_row(), base_row(id=8, answer=20.0)]
    new_rows = [base_row(), base_row(id=8, answer=21.0)]
    (old / "submission.json").write_text(json.dumps(old_rows), encoding="utf-8")
    (new / "submission.json").write_text(json.dumps(new_rows), encoding="utf-8")
    status = artifact_compatibility({"candidate": str(old)}, new)
    assert status["status"] == "row_local"
    assert status["changed_ids"] == [8]
