"""Audit selected source cells for same-report label/context collisions.

The older global collision audit opens every extracted table in every report.
That is useful but unnecessarily expensive after source-cell lineage has been
materialized.  This audit indexes only physical cells that are actually read
by the candidate, then compares different-valued cells with the same or a
near-identical label inside one report.

The result is a review queue, never mutation authority.  A finding still
requires inspection of the original statement and a reproducible calculation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from audit_same_label_value_collisions import (
    collision_priority,
    fold,
    similar_label,
)


def _compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value))


def _report(table_ref: object) -> str:
    value = str(table_ref)
    return value.rsplit("|", 1)[0] if "|" in value else ""


def _physical_key(cell: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(cell.get("source_table", "")),
        int(cell.get("row_idx", -1)),
        int(cell.get("col_idx", -1)),
    )


def _iter_records(payloads: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for payload in payloads:
        yield from payload.get("records", [])


def audit_payloads(
    payloads: Iterable[dict[str, Any]],
    *,
    min_priority: int = 1,
    include_near_labels: bool = True,
    expected_ids: set[int] | None = None,
) -> dict[str, Any]:
    records = list(_iter_records(payloads))
    lineage_ids = {int(record["id"]) for record in records}
    physical: dict[tuple[str, int, int], dict[str, Any]] = {}
    uses: list[dict[str, Any]] = []
    cells_received = 0
    skipped_cells: Counter[str] = Counter()
    skipped_question_ids: dict[str, set[int]] = defaultdict(set)

    for record in records:
        qid = int(record["id"])
        question = str(record.get("question", ""))
        for cell in record.get("cells", []):
            cells_received += 1
            table_ref = str(cell.get("source_table", ""))
            report = _report(table_ref)
            label = str(cell.get("source_label", "")).strip()
            raw = cell.get("raw_physical", cell.get("raw_manifest", ""))
            reason = (
                "missing_report"
                if not report
                else "missing_label"
                if not label
                else "non_numeric_value"
                if not re.search(r"\d", str(raw))
                else ""
            )
            if reason:
                skipped_cells[reason] += 1
                skipped_question_ids[reason].add(qid)
                continue
            key = _physical_key(cell)
            entry = physical.setdefault(
                key,
                {
                    "source_table": table_ref,
                    "report": report,
                    "row_idx": key[1],
                    "col_idx": key[2],
                    "label": label,
                    "label_key": fold(label),
                    "raw": raw,
                    "context": str(cell.get("source_context", "")),
                    "header_path": list(cell.get("header_path", [])),
                    "origin_ids": set(),
                },
            )
            entry["origin_ids"].add(qid)
            if not entry["context"] and cell.get("source_context"):
                entry["context"] = str(cell["source_context"])
            uses.append(
                {
                    "id": qid,
                    "question": question,
                    "answer": record.get("answer"),
                    "physical_key": key,
                }
            )

    by_report_label: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for cell in physical.values():
        by_report_label[cell["report"]][cell["label_key"]].append(cell)

    findings: list[dict[str, Any]] = []
    comparisons = 0
    seen: set[tuple[int, tuple[str, int, int], tuple[str, int, int]]] = set()
    for use in uses:
        current = physical[use["physical_key"]]
        labels = by_report_label[current["report"]]
        candidate_cells = list(labels[current["label_key"]])
        if include_near_labels:
            for label_key, cells in labels.items():
                if label_key == current["label_key"]:
                    continue
                # A question often mentions many operands.  Treating any
                # other selected label that contributes two question words as
                # a competitor floods the queue with intentional numerator,
                # denominator and selector cells.  Near-label mode therefore
                # requires lexical identity of the metric, modulo qualifiers
                # and OCR suffixes; broader context recovery belongs to the
                # full-corpus audit.
                if similar_label(current["label"], label_key):
                    candidate_cells.extend(cells)

        for alternative in candidate_cells:
            alternative_key = (
                alternative["source_table"],
                alternative["row_idx"],
                alternative["col_idx"],
            )
            finding_key = (use["id"], use["physical_key"], alternative_key)
            if finding_key in seen or alternative_key == use["physical_key"]:
                continue
            seen.add(finding_key)
            # When the same question deliberately reads both cells (for a
            # difference, sum, selector or multi-period panel), the second
            # value is an operand rather than a competing source choice.
            if use["id"] in alternative["origin_ids"]:
                continue
            if _compact(current["raw"]) == _compact(alternative["raw"]):
                continue
            comparisons += 1
            priority, reasons = collision_priority(
                use["question"],
                current["context"],
                alternative["context"],
                current["label"],
                alternative["label"],
            )
            if priority < min_priority:
                continue
            findings.append(
                {
                    "id": use["id"],
                    "priority": priority,
                    "reasons": reasons,
                    "question": use["question"],
                    "answer": use["answer"],
                    "current": {
                        **{k: v for k, v in current.items() if k != "origin_ids"},
                        "origin_ids": sorted(current["origin_ids"]),
                    },
                    "alternative": {
                        **{k: v for k, v in alternative.items() if k != "origin_ids"},
                        "origin_ids": sorted(alternative["origin_ids"]),
                    },
                }
            )

    findings.sort(
        key=lambda row: (
            -int(row["priority"]),
            int(row["id"]),
            row["current"]["source_table"],
            row["alternative"]["source_table"],
        )
    )
    usable_ids = {use["id"] for use in uses}
    expected = expected_ids or lineage_ids
    return {
        "kind": "selected_source_cell_collision_review_queue",
        "records_received": len(records),
        "lineage_question_count": len(lineage_ids),
        "expected_question_count": len(expected),
        "missing_lineage_question_ids": sorted(expected - lineage_ids),
        "question_count_with_usable_numeric_cells": len(usable_ids),
        "question_ids_without_usable_numeric_cells": sorted(expected - usable_ids),
        "source_cells_received": cells_received,
        "source_cell_uses": len(uses),
        "skipped_cell_counts": dict(sorted(skipped_cells.items())),
        "skipped_cell_question_ids": {
            reason: sorted(ids) for reason, ids in sorted(skipped_question_ids.items())
        },
        "unique_physical_cells": len(physical),
        "reports_indexed": len(by_report_label),
        "different_value_comparisons": comparisons,
        "finding_count": len(findings),
        "finding_question_count": len({row["id"] for row in findings}),
        "findings": findings,
        "policy": (
            "Read-only triage. Verify the exact original statement, scope, period, "
            "unit and terminal calculation before changing any submission row."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-priority", type=int, default=1)
    parser.add_argument("--exact-labels-only", action="store_true")
    parser.add_argument(
        "--submission",
        type=Path,
        help="optional submission directory or submission.json used for coverage IDs",
    )
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.lineage]
    expected_ids: set[int] | None = None
    if args.submission:
        submission_path = (
            args.submission / "submission.json"
            if args.submission.is_dir()
            else args.submission
        )
        expected_ids = {
            int(row["id"])
            for row in json.loads(submission_path.read_text(encoding="utf-8-sig"))
        }
    result = audit_payloads(
        payloads,
        min_priority=args.min_priority,
        include_near_labels=not args.exact_labels_only,
        expected_ids=expected_ids,
    )
    result["lineage"] = [str(path.resolve()) for path in args.lineage]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "findings"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
