from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ingest_v275_residual_adjudication import build_entries


def _row(qid: int, status: str, current: float, recomputed: float) -> dict:
    return {
        "id": qid,
        "status": status,
        "current_answer": current,
        "recomputed_answer": recomputed,
        "proof": "physical source and arithmetic independently checked",
        "physical_refs": [{"table_ref": f"AAA|{qid}", "row_idx": 1, "col_idx": 2, "raw": "12"}],
        "hard_negatives": [{"why_wrong": "prior-year column"}],
        "proposed_mutation": {"answer": recomputed} if status != "source_confirmed_no_change" else None,
    }


def _report(path: Path) -> dict:
    payload = {
        "structurally_valid": True,
        "ready_for_terminal_ledger_ingest": True,
        "records": [
            _row(1, "source_confirmed_no_change", 12, 12),
            _row(2, "source_confirmed_change", 10, 11),
            _row(3, "cleanup_only", 9, 9),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_build_entries_separates_change_cleanup_and_keep(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = _report(path)
    entries = build_entries(
        report,
        {"verdicts": {}},
        approved_change_ids={2},
        approved_cleanup_ids={3},
        report_path=path,
    )
    assert entries["1"]["mutation"] == "none"
    assert entries["2"]["old_answer"] == 10
    assert entries["2"]["answer"] == 11
    assert entries["3"]["mutation"] == "pending_batch_cleanup"


def test_change_requires_exact_parent_approval(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = _report(path)
    with pytest.raises(ValueError, match="answer-change approval mismatch"):
        build_entries(
            report,
            {"verdicts": {}},
            approved_change_ids=set(),
            approved_cleanup_ids={3},
            report_path=path,
        )


def test_existing_ledger_entry_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = _report(path)
    with pytest.raises(ValueError, match="already exists"):
        build_entries(
            report,
            {"verdicts": {"1": {"status": "old"}}},
            approved_change_ids={2},
            approved_cleanup_ids={3},
            report_path=path,
        )

