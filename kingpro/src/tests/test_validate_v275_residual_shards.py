from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_v275_residual_shards import sha256, validate


def _fixture(tmp_path: Path, *, status: str = "source_confirmed_no_change") -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "submission.json").write_text(
        json.dumps([{"id": 1, "answer": 12.0}]), encoding="utf-8"
    )
    record = {
        "id": 1,
        "status": status,
        "current_answer": 12.0,
        "recomputed_answer": 13.0 if status == "source_confirmed_change" else 12.0,
        "proof": "physical row and unit checked",
        "physical_refs": ["AAA|1"],
        "hard_negatives": ["prior-year column rejected"],
        "proposed_mutation": {"answer": 13.0} if status == "source_confirmed_change" else "none",
    }
    shard = tmp_path / "shard.json"
    shard.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    manifest = {
        "candidate": "candidate",
        "candidate_submission_sha256": sha256(candidate / "submission.json"),
        "allowed_statuses": [
            "source_confirmed_no_change",
            "source_confirmed_change",
            "cleanup_only",
            "ambiguous",
        ],
        "required_record_fields": list(record),
        "shards": {"a": {"output": "shard.json", "ids": [1]}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_no_change_shard(tmp_path: Path) -> None:
    report = validate(_fixture(tmp_path), root=tmp_path)
    assert report["structurally_valid"] is True
    assert report["ready_for_terminal_ledger_ingest"] is True
    assert report["status_counts"] == {"source_confirmed_no_change": 1}


def test_valid_answer_change_is_separated(tmp_path: Path) -> None:
    report = validate(
        _fixture(tmp_path, status="source_confirmed_change"), root=tmp_path
    )
    assert report["structurally_valid"] is True
    assert report["answer_change_ids"] == [1]


def test_missing_hard_negative_fails_closed(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    shard = json.loads((tmp_path / "shard.json").read_text(encoding="utf-8"))
    shard["records"][0]["hard_negatives"] = []
    (tmp_path / "shard.json").write_text(json.dumps(shard), encoding="utf-8")
    report = validate(manifest, root=tmp_path)
    assert report["structurally_valid"] is False
    assert any(item["kind"] == "hard-negatives-empty" for item in report["issues"])


def test_agent_alias_fields_are_normalized(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path)
    shard = json.loads((tmp_path / "shard.json").read_text(encoding="utf-8"))
    row = shard["records"][0]
    row["source_refs"] = row.pop("physical_refs")
    row["hard_negative"] = row.pop("hard_negatives")[0]
    (tmp_path / "shard.json").write_text(json.dumps(shard), encoding="utf-8")
    report = validate(manifest, root=tmp_path)
    assert report["structurally_valid"] is True
    assert report["records"][0]["physical_refs"] == ["AAA|1"]
    assert report["records"][0]["hard_negatives"] == ["prior-year column rejected"]
