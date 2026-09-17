from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_cross_task_risk_matrix.py"
SPEC = importlib.util.spec_from_file_location("audit_cross_task_risk_matrix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_reason_features_retrieve_failure_families() -> None:
    assert module.failure_modes_for_reasons([
        "deep_or_noisy_header",
        "subtotal_component_overlap",
    ]) == [
        "additivity_fanout_and_duplicate_rows",
        "hard_negative_metric_or_table",
        "hierarchical_headers_and_merged_cells",
    ]


def test_review_memory_indexes_each_question(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    events = [
        {
            "id": "R1",
            "kind": "review",
            "question_ids": [69, 588],
            "verdict": "source_confirmed",
        },
        {"id": "E1", "kind": "experiment", "question_ids": [69]},
        {
            "id": "R2",
            "kind": "review",
            "question_ids": [69],
            "verdict": "candidate_fix",
        },
    ]
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
    )
    memory = module.load_review_memory(path)
    assert [item["id"] for item in memory[69]] == ["R1", "R2"]
    assert [item["id"] for item in memory[588]] == ["R1"]


def test_missing_memory_file_is_safe(tmp_path: Path) -> None:
    assert module.load_review_memory(tmp_path / "missing.jsonl") == {}


def test_source_backed_false_positive_is_already_verified() -> None:
    assert module.is_source_verified_review({
        "verdict": "false_positive",
        "oracle": "source",
        "oracle_trust": "that",
    })


def test_candidate_and_ambiguous_reviews_stay_reviewable() -> None:
    assert not module.is_source_verified_review({
        "verdict": "candidate_fix",
        "oracle": "source",
        "oracle_trust": "that",
    })
    assert not module.is_source_verified_review({
        "verdict": "ambiguous_keep_baseline",
        "oracle": "source",
        "oracle_trust": "that",
    })


def test_historical_source_oracle_spelling_is_supported() -> None:
    assert module.is_source_verified_review({
        "verdict": "source-confirmed-no-change",
        "oracle": "exact-source-cells",
        "oracle_trust": "khong-ro",
    })
