"""Ingest the parent-approved, fail-closed V275 residual adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "build" / "v275_residual_source_adjudication.json"
DEFAULT_LEDGER = ROOT / "knowledge" / "vothuong" / "question_source_verdicts.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def compact_ref(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    return {
        key: item[key]
        for key in (
            "role",
            "table_ref",
            "row_idx",
            "col_idx",
            "raw",
            "source_unit",
            "resolved_path",
            "csv_sha256",
        )
        if key in item
    }


def proof_text(proof: Any) -> str:
    if isinstance(proof, str):
        return proof.strip()
    if not isinstance(proof, dict):
        return str(proof)
    parts = []
    for key in (
        "company_check",
        "year_check",
        "scope_check",
        "formula",
        "exact_unrounded",
        "rounding",
        "unit_check",
    ):
        if key in proof:
            parts.append(f"{key}={proof[key]}")
    return "; ".join(parts)


def build_entries(
    report: dict[str, Any],
    ledger: dict[str, Any],
    *,
    approved_change_ids: set[int],
    approved_cleanup_ids: set[int],
    report_path: Path,
) -> dict[str, dict[str, Any]]:
    if report.get("structurally_valid") is not True:
        raise ValueError("merged report is not structurally valid")
    if report.get("ready_for_terminal_ledger_ingest") is not True:
        raise ValueError("merged report is not terminal-ready")
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("merged report has no records")
    actual_changes = {
        int(row["id"])
        for row in records
        if row.get("status") == "source_confirmed_change"
    }
    actual_cleanups = {
        int(row["id"]) for row in records if row.get("status") == "cleanup_only"
    }
    if actual_changes != approved_change_ids:
        raise ValueError(
            f"answer-change approval mismatch: report={sorted(actual_changes)} approved={sorted(approved_change_ids)}"
        )
    if actual_cleanups != approved_cleanup_ids:
        raise ValueError(
            f"cleanup approval mismatch: report={sorted(actual_cleanups)} approved={sorted(approved_cleanup_ids)}"
        )
    verdicts = ledger.get("verdicts")
    if not isinstance(verdicts, dict):
        raise ValueError("ledger has no verdicts object")
    staged: dict[str, dict[str, Any]] = {}
    try:
        artifact = str(report_path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        artifact = str(report_path.resolve()).replace("\\", "/")
    artifact_sha = sha256(report_path)
    for row in records:
        qid = int(row["id"])
        if str(qid) in verdicts:
            raise ValueError(f"q{qid} already exists in ledger")
        status = str(row["status"])
        if status == "source_confirmed_change":
            ledger_status = "source_confirmed_answer_change_via_v275_residual"
            mutation = "pending_batch_answer_source_retrieval_fix"
        elif status == "cleanup_only":
            ledger_status = "source_confirmed_pending_batch_cleanup_via_v275_residual"
            mutation = "pending_batch_cleanup"
        elif status == "source_confirmed_no_change":
            ledger_status = "source_confirmed_no_change_via_v275_residual"
            mutation = "none"
        else:
            raise ValueError(f"q{qid}: unsupported status {status!r}")
        refs = row.get("physical_refs")
        negatives = row.get("hard_negatives")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"q{qid}: physical refs missing")
        if not isinstance(negatives, list) or not negatives:
            raise ValueError(f"q{qid}: hard negatives missing")
        proof = proof_text(row.get("proof"))
        if len(proof) < 12:
            raise ValueError(f"q{qid}: proof too short")
        entry: dict[str, Any] = {
            "status": ledger_status,
            "answer": row.get("recomputed_answer"),
            "confidence": "high",
            "proof": proof,
            "source_refs": [compact_ref(item) for item in refs],
            "hard_negatives": negatives,
            "review_artifact": artifact,
            "review_artifact_sha256": artifact_sha,
            "proposed_mutation": row.get("proposed_mutation"),
            "mutation": mutation,
        }
        if status == "source_confirmed_change":
            entry["old_answer"] = row.get("current_answer")
        if isinstance(row.get("proof"), dict):
            entry["proof_details"] = row["proof"]
        staged[str(qid)] = entry
    return staged


def parse_ids(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--approve-answer-changes", default="")
    parser.add_argument("--approve-cleanups", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report_path = args.report.resolve()
    ledger_path = args.ledger.resolve()
    report = load(report_path)
    ledger = load(ledger_path)
    staged = build_entries(
        report,
        ledger,
        approved_change_ids=parse_ids(args.approve_answer_changes),
        approved_cleanup_ids=parse_ids(args.approve_cleanups),
        report_path=report_path,
    )
    if not args.dry_run:
        ledger["verdicts"].update(staged)
        ledger_path.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "ingested_count": len(staged),
                "ids": [int(value) for value in staged],
                "answer_change_ids": sorted(parse_ids(args.approve_answer_changes)),
                "cleanup_ids": sorted(parse_ids(args.approve_cleanups)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
