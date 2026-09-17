"""Find source cells displaced by malformed rowspan/colspan expansion.

Some extracted financial tables expand a two-column year header to five
physical columns while a data row puts the current-period value one column to
the left of that header.  The comparative value is then duplicated across the
nominal current- and prior-period columns.  A selector that trusts the header
column reads last year's value even though the table visibly contains the
current value.

This audit consumes the physical cell-lineage report and emits only a narrow,
high-confidence pattern:

* the selected column's header contains the requested year;
* the selected scalar is duplicated immediately to its right; and
* the scalar immediately to its left is different.

It is a read-only triage tool.  Every finding still requires checking the full
source table and recomputing the question before a candidate is built.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from audit_source_cell_semantics import header_path, is_financial_number, parsed_table


ROOT = Path(__file__).resolve().parents[1]


def canonical_number(value: object) -> str | None:
    text = str(value).strip()
    if not is_financial_number(text):
        return None
    negative = text.startswith("(") and text.endswith(")")
    compact = re.sub(r"[().,%\s]", "", text)
    compact = compact.lstrip("+")
    if negative and not compact.startswith("-"):
        compact = "-" + compact
    return compact


def looks_like_accounting_amount(value: object) -> bool:
    canonical = canonical_number(value)
    if canonical is None:
        return False
    digits = re.sub(r"\D", "", canonical)
    return len(digits) >= 5


def shifted_left_candidate(
    frame: Any,
    row: int,
    column: int,
    target_year: object,
) -> dict[str, Any] | None:
    if row < 0 or row >= len(frame) or column <= 0 or column + 1 >= len(frame.columns):
        return None
    year = str(target_year).strip()
    selected_headers = header_path(frame, column, row)
    candidate_headers = header_path(frame, column - 1, row)
    if not year or not any(year in str(value) for value in selected_headers):
        return None
    # If the left column is itself explicitly dated to the requested year,
    # this is usually a legitimate adjacent scope (for example Group versus
    # Parent Company), not a malformed header shift.
    if any(year in str(value) for value in candidate_headers):
        return None

    left_raw = str(frame.iloc[row, column - 1]).strip()
    selected_raw = str(frame.iloc[row, column]).strip()
    right_raw = str(frame.iloc[row, column + 1]).strip()
    left = canonical_number(left_raw)
    selected = canonical_number(selected_raw)
    right = canonical_number(right_raw)
    if left is None or selected is None or right is None:
        return None
    if selected != right or left == selected:
        return None

    return {
        "confidence": "high",
        "requested_year": year,
        "selected_column": column,
        "candidate_column": column - 1,
        "selected_header_path": selected_headers,
        "candidate_header_path": candidate_headers,
        "selected_raw": selected_raw,
        "right_duplicate_raw": right_raw,
        "candidate_raw": left_raw,
    }


def undated_left_amount_candidate(
    frame: Any,
    row: int,
    column: int,
    target_year: object,
) -> dict[str, Any] | None:
    """Broader review hint when a large amount sits under an undated left cell."""

    if row < 0 or row >= len(frame) or column <= 0 or column >= len(frame.columns):
        return None
    year = str(target_year).strip()
    selected_headers = header_path(frame, column, row)
    candidate_headers = header_path(frame, column - 1, row)
    if not year or not any(year in str(value) for value in selected_headers):
        return None
    if any(year in str(value) for value in candidate_headers):
        return None
    left_raw = str(frame.iloc[row, column - 1]).strip()
    selected_raw = str(frame.iloc[row, column]).strip()
    if not looks_like_accounting_amount(left_raw) or not looks_like_accounting_amount(selected_raw):
        return None
    if canonical_number(left_raw) == canonical_number(selected_raw):
        return None
    return {
        "confidence": "review",
        "requested_year": year,
        "selected_column": column,
        "candidate_column": column - 1,
        "selected_header_path": selected_headers,
        "candidate_header_path": candidate_headers,
        "selected_raw": selected_raw,
        "candidate_raw": left_raw,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.lineage.read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    scanned = 0
    for record in payload.get("records", []):
        qid = int(record["id"])
        for cell in record.get("cells", []):
            scanned += 1
            try:
                frame = parsed_table(str(cell["source_table"]))
                detected = shifted_left_candidate(
                    frame,
                    int(cell["row_idx"]),
                    int(cell["col_idx"]),
                    cell.get("year", ""),
                )
                if detected is None:
                    detected = undated_left_amount_candidate(
                        frame,
                        int(cell["row_idx"]),
                        int(cell["col_idx"]),
                        cell.get("year", ""),
                    )
            except Exception as error:
                errors.append({
                    "id": qid,
                    "source_table": cell.get("source_table"),
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            if detected:
                findings.append({
                    "id": qid,
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "metric_key": cell.get("metric_key"),
                    "source_table": cell.get("source_table"),
                    "row_idx": cell.get("row_idx"),
                    "source_label": cell.get("source_label"),
                    **detected,
                })

    high_findings = [item for item in findings if item["confidence"] == "high"]
    review_findings = [item for item in findings if item["confidence"] == "review"]
    question_ids = sorted({int(item["id"]) for item in findings})
    result = {
        "lineage": str(args.lineage),
        "physical_cells_scanned": scanned,
        "finding_count": len(findings),
        "high_confidence_count": len(high_findings),
        "review_hint_count": len(review_findings),
        "question_count": len(question_ids),
        "question_ids": question_ids,
        "resolution_error_count": len(errors),
        "policy": "High-confidence merged-header shift hints only; verify full source and recompute before repair.",
        "findings": findings,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "physical_cells_scanned",
        "finding_count",
        "high_confidence_count",
        "review_hint_count",
        "question_count",
        "question_ids",
        "resolution_error_count",
    )}, ensure_ascii=False, indent=2))
    print("output", args.out)


if __name__ == "__main__":
    main()
