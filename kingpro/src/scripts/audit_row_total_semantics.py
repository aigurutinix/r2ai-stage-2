"""Flag possible row-subtotal versus grand-total mistakes.

The column audit catches a value taken from the wrong dimensional column.  It
cannot catch the orthogonal failure mode where the correct column is used but
the program stops at a component/subtotal row while the question asks for a
total.  This script builds a conservative review queue from compact source
cells.  It never changes a submission.

The queue is deliberately evidence-rich: it records the selected row, every
explicit total row in the same physical table, and whether the selected row is
itself a verified local subtotal of the contiguous indented rows below it.
That last check prevents a bare ``TONG CONG`` at the end of a multi-section
note from overriding the exact subtotal named in the question.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_legacy_source_scope import fold
from audit_total_bucket_semantics import (
    financial_number,
    load_catalog_paths,
    read_csv,
    resolve_compact_source_row,
)


TOTAL_QUESTION_RE = re.compile(
    r"\b(tong|toan bo|tat ca)\b"
)
TOTAL_ROW_RE = re.compile(
    r"^(?:tong(?: cong| so)?|cong)$|\bgrand total\b"
)
CHILD_ROW_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")


def asks_for_total(question: object) -> bool:
    return bool(TOTAL_QUESTION_RE.search(fold(question)))


def is_total_row(label: object) -> bool:
    return bool(TOTAL_ROW_RE.search(fold(label).strip()))


def is_child_label(label: object) -> bool:
    return bool(CHILD_ROW_RE.match(str(label)))


def contiguous_child_values(
    rows: list[list[str]], source_row: int, source_column: int
) -> list[dict[str, Any]]:
    """Return numeric, visibly indented rows immediately below a source row."""

    children: list[dict[str, Any]] = []
    for index in range(source_row + 1, len(rows)):
        row = rows[index]
        label = row[0] if row else ""
        if not is_child_label(label):
            break
        raw = row[source_column] if source_column < len(row) else ""
        value = financial_number(raw)
        if value is not None:
            children.append(
                {"row": index, "label": label, "raw": raw, "value": str(value)}
            )
    return children


def local_subtotal_evidence(
    rows: list[list[str]], source_row: int, source_column: int
) -> dict[str, Any]:
    selected_raw = rows[source_row][source_column]
    selected = financial_number(selected_raw)
    children = contiguous_child_values(rows, source_row, source_column)
    child_values = [Decimal(child["value"]) for child in children]
    child_sum = sum(child_values, Decimal(0)) if child_values else None
    return {
        "selected_value": str(selected) if selected is not None else None,
        "contiguous_children": children,
        "child_sum": str(child_sum) if child_sum is not None else None,
        "selected_equals_child_sum": bool(
            selected is not None and child_sum is not None and selected == child_sum
        ),
    }


def explicit_total_rows(
    rows: list[list[str]], source_column: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not row or source_column >= len(row) or not is_total_row(row[0]):
            continue
        value = financial_number(row[source_column])
        if value is None:
            continue
        result.append(
            {
                "row": index,
                "label": row[0],
                "raw": row[source_column],
                "value": str(value),
            }
        )
    return result


def build_queue(submission: Path) -> dict[str, Any]:
    catalog = load_catalog_paths()
    records = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    table_cache: dict[str, list[list[str]]] = {}
    findings: list[dict[str, Any]] = []
    reads_checked = 0
    resolution_errors: list[dict[str, Any]] = []

    for record in records:
        if not asks_for_total(record.get("question", "")):
            continue
        for evidence in record.get("evidence", []):
            compact_path = submission / str(evidence.get("csv_path", ""))
            if not compact_path.is_file():
                continue
            with compact_path.open(
                encoding="utf-8-sig", errors="replace", newline=""
            ) as handle:
                sources = list(csv.DictReader(handle))
            required = {"source_table", "row_idx", "col_idx", "raw"}
            if not sources or not required.issubset(sources[0]):
                continue
            for source in sources:
                table_ref = str(source.get("source_table", ""))
                csv_path = catalog.get(table_ref)
                if csv_path is None or not csv_path.is_file():
                    resolution_errors.append(
                        {
                            "id": int(record["id"]),
                            "source_table": table_ref,
                            "reason": "catalog table CSV not found",
                        }
                    )
                    continue
                if table_ref not in table_cache:
                    table_cache[table_ref] = read_csv(csv_path)
                rows = table_cache[table_ref]
                reads_checked += 1
                source_row = resolve_compact_source_row(rows, source)
                try:
                    source_column = int(source.get("col_idx", -1))
                except (TypeError, ValueError):
                    source_column = -1
                if (
                    source_row is None
                    or source_column <= 0
                    or source_column >= len(rows[source_row])
                ):
                    resolution_errors.append(
                        {
                            "id": int(record["id"]),
                            "source_table": table_ref,
                            "raw": source.get("raw"),
                            "reason": "source cell not resolved",
                        }
                    )
                    continue
                selected_label = rows[source_row][0] if rows[source_row] else ""
                if is_total_row(selected_label):
                    continue
                totals = explicit_total_rows(rows, source_column)
                selected_value = financial_number(rows[source_row][source_column])
                differing_totals = [
                    total
                    for total in totals
                    if selected_value is None
                    or Decimal(total["value"]) != selected_value
                ]
                if not differing_totals:
                    continue
                local = local_subtotal_evidence(rows, source_row, source_column)
                row_path = [
                    {
                        "row": index,
                        "label": row[0] if row else "",
                        "raw": row[source_column] if source_column < len(row) else "",
                    }
                    for index, row in enumerate(rows)
                    if row
                    and (
                        (source_column < len(row) and financial_number(row[source_column]) is not None)
                        or str(row[0]).strip()
                    )
                ]
                findings.append(
                    {
                        "id": int(record["id"]),
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "source_table": table_ref,
                        "source_row": source_row,
                        "source_column": source_column,
                        "selected_label": selected_label,
                        "selected_raw": rows[source_row][source_column],
                        "explicit_total_rows": differing_totals,
                        "local_subtotal_evidence": local,
                        "physical_row_path": row_path,
                        "priority": (
                            "low_local_subtotal_confirmed"
                            if local["selected_equals_child_sum"]
                            else "review"
                        ),
                    }
                )

    findings.sort(
        key=lambda item: (
            item["priority"] == "low_local_subtotal_confirmed",
            int(item["id"]),
        )
    )
    return {
        "kind": "row_subtotal_vs_grand_total_review_queue",
        "submission": str(submission.resolve()),
        "reads_checked": reads_checked,
        "resolution_error_count": len(resolution_errors),
        "finding_count": len(findings),
        "question_count": len({item["id"] for item in findings}),
        "review_count": sum(item["priority"] == "review" for item in findings),
        "local_subtotal_confirmed_count": sum(
            item["priority"] == "low_local_subtotal_confirmed"
            for item in findings
        ),
        "findings": findings,
        "resolution_errors": resolution_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_queue(args.submission.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "findings"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
