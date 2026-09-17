import json
from pathlib import Path

from scripts.build_independent_batch_risk_queue import build_queue


def test_queue_requires_independent_families_and_penalizes_review(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    reports = tmp_path / "reports"
    candidate.mkdir()
    reports.mkdir()
    (candidate / "submission.json").write_text(
        json.dumps(
            [
                {"id": 1, "question": "q1", "answer": 1, "relevant_tables": []},
                {"id": 2, "question": "q2", "answer": 2, "relevant_tables": []},
                {"id": 3, "question": "q3", "answer": 3, "relevant_tables": []},
            ]
        ),
        encoding="utf-8",
    )
    empty = {"findings": []}
    for filename in set(__import__(
        "scripts.build_independent_batch_risk_queue", fromlist=["REPORTS"]
    ).REPORTS.values()):
        (reports / filename).write_text(json.dumps(empty), encoding="utf-8")
    (reports / "v210_batch_program_intent_all_v209.json").write_text(
        json.dumps({"findings": [{"id": 1, "score": 6}, {"id": 2, "score": 6}]}),
        encoding="utf-8",
    )
    (reports / "v210_batch_semantic_alignment.json").write_text(
        json.dumps({"findings": [{"id": 1, "score": 0}, {"id": 2, "score": 0}, {"id": 3, "score": 0}]}),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "kind": "review",
                "id": "R1",
                "question_ids": [1],
                "verdict": "source_confirmed",
                "summary": "checked",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_queue(candidate, reports, ledger, 10)

    assert [row["id"] for row in payload["queue"]] == [2, 1]
    assert payload["eligible_multi_signal_questions"] == 2
    assert payload["queue"][1]["strong_source_review_count"] == 1


def test_q769_source_proven_bonus_keeps_it_first(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    reports = tmp_path / "reports"
    candidate.mkdir()
    reports.mkdir()
    (candidate / "submission.json").write_text(
        json.dumps(
            [
                {"id": 5, "question": "q5", "answer": 5, "relevant_tables": []},
                {"id": 769, "question": "q769", "answer": 26.89, "relevant_tables": []},
            ]
        ),
        encoding="utf-8",
    )
    module = __import__("scripts.build_independent_batch_risk_queue", fromlist=["REPORTS"])
    for filename in set(module.REPORTS.values()):
        (reports / filename).write_text(json.dumps({"findings": []}), encoding="utf-8")
    (reports / "v210_operand_table_context_consistency_v209.json").write_text(
        json.dumps({"findings": [{"id": 5}, {"id": 769}]}), encoding="utf-8"
    )
    (reports / "v210_batch_semantic_alignment.json").write_text(
        json.dumps({"findings": [{"id": 5, "score": 0}, {"id": 769, "score": 0}]}),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")

    payload = build_queue(candidate, reports, ledger, 2)

    assert payload["queue"][0]["id"] == 769
    assert payload["queue"][0]["status"] == "source_proven_repair"
