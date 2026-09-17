from scripts.render_compact_audit_tasks import compact_task


def test_compact_task_drops_large_runtime_namespace_and_limits_failures() -> None:
    record = {
        "id": 1,
        "question": "q",
        "answer": 2.0,
        "runtime": {"string": {"status": "pass", "result": 2.0, "scalars": {"v0": 1}}},
        "physical_audit": {"status": "failure", "field_failures": [1, 2, 3, 4]},
        "unused_evidence": None,
    }

    task = compact_task(record)

    assert "scalars" not in task["runtime"]["string"]
    assert task["physical"]["failures"] == [1, 2, 3]
    assert task["instructions"]["chat_response"].startswith("Return only")
