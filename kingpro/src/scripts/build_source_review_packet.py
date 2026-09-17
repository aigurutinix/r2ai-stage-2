"""Build compact, source-bound review packets for batches of submission rows.

The project has two provenance formats:

* deterministic builders record rows in ``source_audit.json`` or
  ``panel_source_audit.json``; and
* legacy queries are resolved by ``audit_legacy_query_sources.py``.

This tool joins both formats with the submission registry and nearby report
context.  It is deliberately read-only: the packet accelerates manual review
but never changes an answer or claims that a source cell is semantically gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from audit_legacy_source_scope import csv_header, nearby_context, report_index


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def parse_ids(text: str) -> list[int]:
    """Parse comma/space-separated IDs while preserving first-seen order."""

    result: list[int] = []
    seen: set[int] = set()
    for token in text.replace(",", " ").split():
        value = int(token)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def query_digest(query: str) -> dict[str, Any]:
    lines = [line.strip() for line in str(query or "").splitlines()]
    comments = [line for line in lines if line.startswith("#")]
    result_lines = [line for line in lines if line.startswith("result") and "=" in line]
    return {
        "comments": comments,
        "result_expressions": result_lines,
        "result_expression": result_lines[-1] if result_lines else "",
    }


def _context(table_ref: str, reports: dict[str, Path]) -> str:
    if "|" not in table_ref:
        return ""
    document, line_text = table_ref.rsplit("|", 1)
    report = reports.get(document)
    if report is None:
        return ""
    try:
        return nearby_context(report, int(line_text))
    except ValueError:
        return ""


def _audit_index(submission: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for name in ("source_audit.json", "panel_source_audit.json"):
        for row in load_json(submission / name, []) or []:
            if isinstance(row, dict) and "id" in row:
                result[int(row["id"])] = row
    return result


def _row_fingerprint(row: dict[str, Any] | None) -> str:
    """Fingerprint every field that can change terminal reads or their scope."""

    if not isinstance(row, dict):
        return ""
    payload = {
        "question": row.get("question"),
        "answer": row.get("answer"),
        "relevant_docs": row.get("relevant_docs") or [],
        "relevant_tables": row.get("relevant_tables") or [],
        "pandas_query": row.get("pandas_query"),
        "evidence": row.get("evidence") or [],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_packet(
    submission: Path,
    legacy_audit: Path,
    ids: list[int],
) -> dict[str, Any]:
    registry = {
        int(row["id"]): row
        for row in load_json(submission / "submission.json", []) or []
        if isinstance(row, dict) and "id" in row
    }
    legacy_payload = load_json(legacy_audit, {}) or {}
    legacy = {
        int(row["id"]): row
        for row in legacy_payload.get("records", [])
        if isinstance(row, dict) and "id" in row
    }
    audited_submission_text = str(legacy_payload.get("submission") or "")
    audited_submission = Path(audited_submission_text) if audited_submission_text else None
    audited_registry: dict[int, dict[str, Any]] = {}
    if audited_submission and (audited_submission / "submission.json").is_file():
        audited_registry = {
            int(row["id"]): row
            for row in load_json(audited_submission / "submission.json", []) or []
            if isinstance(row, dict) and "id" in row
        }
    same_submission = bool(
        audited_submission and audited_submission.resolve() == submission.resolve()
    )
    deterministic = _audit_index(submission)
    reports = report_index()
    records: list[dict[str, Any]] = []
    missing_ids: list[int] = []
    missing_provenance: list[int] = []
    stale_legacy_ids: list[int] = []

    for question_id in ids:
        row = registry.get(question_id)
        if row is None:
            missing_ids.append(question_id)
            continue
        sources: list[dict[str, Any]] = []
        provenance = "unresolved"

        if question_id in deterministic:
            provenance = "deterministic_source_audit"
            audit = deterministic[question_id]
            for source in audit.get("sources", []):
                table_ref = str(source.get("table_ref", ""))
                sources.append({
                    "table_ref": table_ref,
                    "source_label": source.get("label", ""),
                    "source_row_labels": source.get("source_row_labels", []),
                    "raw": source.get("raw", ""),
                    "row": source.get("row"),
                    "column": source.get("column"),
                    "scale": source.get("scale"),
                    "typed_factor": source.get("typed_factor"),
                    "metric": source.get("metric", ""),
                    "context": _context(table_ref, reports),
                })
            provenance_note = audit.get("note", "")
            audit_answer = audit.get("answer")
        elif question_id in legacy and (
            same_submission
            or _row_fingerprint(row) == _row_fingerprint(audited_registry.get(question_id))
        ):
            provenance = "legacy_terminal_read"
            audit = legacy[question_id]
            for source in audit.get("terminal_reads", []):
                table_ref = str(source.get("source_table", ""))
                sources.append({
                    "table_ref": table_ref,
                    "source_label": source.get("source_label", ""),
                    "source_row_values": source.get("source_row_values", []),
                    "raw": source.get("raw", ""),
                    "row": source.get("source_row"),
                    "column": source.get("source_column"),
                    "column_label": source.get("source_column_label", ""),
                    "access_kind": source.get("access_kind", ""),
                    "source_header": csv_header(submission, str(source.get("csv", "")), rows=3),
                    "context": _context(table_ref, reports),
                })
            provenance_note = "Resolved from the terminal dataframe reads used by the submitted program."
            audit_answer = audit.get("answer")
        elif question_id in legacy:
            provenance = "stale_legacy_audit"
            provenance_note = (
                "Rejected legacy terminal reads because the audit was produced for "
                "a different submission and this row's execution/scope fingerprint changed."
            )
            audit_answer = None
            missing_provenance.append(question_id)
            stale_legacy_ids.append(question_id)
        else:
            provenance_note = "No deterministic or legacy exact-cell record found."
            audit_answer = None
            missing_provenance.append(question_id)

        records.append({
            "id": question_id,
            "question": row.get("question", ""),
            "answer": row.get("answer"),
            "relevant_docs": row.get("relevant_docs", []),
            "relevant_tables": row.get("relevant_tables", []),
            "provenance": provenance,
            "provenance_note": provenance_note,
            "audit_answer": audit_answer,
            "answer_matches_audit": audit_answer == row.get("answer") if audit_answer is not None else None,
            "query": query_digest(str(row.get("pandas_query", ""))),
            "sources": sources,
            "review_status": "pending_source_semantic_review",
        })

    return {
        "schema_version": 1,
        "submission": str(submission.resolve()),
        "legacy_audit": str(legacy_audit.resolve()),
        "legacy_audit_submission": (
            str(audited_submission.resolve()) if audited_submission else None
        ),
        "legacy_audit_same_submission": same_submission,
        "requested_ids": ids,
        "record_count": len(records),
        "missing_ids": missing_ids,
        "missing_provenance": missing_provenance,
        "stale_legacy_ids": stale_legacy_ids,
        "automatic_answer_changes": False,
        "records": records,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("legacy_audit", type=Path)
    parser.add_argument("--ids", required=True, help="Comma/space-separated question IDs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = build_packet(args.submission_dir, args.legacy_audit, parse_ids(args.ids))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
