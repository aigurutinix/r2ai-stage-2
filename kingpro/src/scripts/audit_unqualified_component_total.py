"""Find unqualified metrics bound to one qualified component instead of a total.

This is the broader companion to ``audit_total_row_component_semantics``.  It
does not require the literal word ``tổng``.  Instead, it looks for a selected
row carrying an explicit component qualifier (geography, maturity, provision
type, currency, or counterparty) that the metric question does not request.
It then requires sibling rows from the same qualifier family and an exact
additive blank/total row in the same physical column.

Findings are read-only review candidates.  The strict family/core/additivity
gates deliberately trade recall for source-level precision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from audit_legacy_source_scope import fold
from audit_source_cell_semantics import parsed_table, physical_label
from audit_total_row_component_semantics import (
    financial_number,
    finding_is_covered_by_selected_cells,
    label_is_total,
)


QUALIFIER_FAMILIES: dict[str, tuple[str, ...]] = {
    "geography": (
        r"tai cac thi truong nuoc ngoai",
        r"tai thi truong nuoc ngoai",
        r"tai viet nam",
        r"trong nuoc",
        r"nuoc ngoai",
        r"noi dia",
        r"xuat khau",
    ),
    "maturity": (
        r"ngan han",
        r"trung han",
        r"dai han",
        r"duoi mot nam",
        r"duoi 1 nam",
        r"tren mot nam",
        r"tren 1 nam",
    ),
    "provision_type": (
        r"du phong chung",
        r"du phong cu the",
        r"du phong giam gia",
    ),
    "currency": (
        r"bang vnd",
        r"bang usd",
        r"bang ngoai te",
        r"bang dong viet nam",
    ),
    "counterparty": (
        r"ben thu ba",
        r"ben lien quan",
        r"cong ty con",
        r"cong ty lien ket",
    ),
}

CORE_STOPWORDS = {
    "tai", "cac", "thi", "truong", "bang", "va", "cua", "theo", "so",
    "cuoi", "dau", "nam", "ky", "gia", "tri", "trieu", "ty", "dong",
    "vnd", "usd", "thuyet", "minh",
}


def matched_family(label: object) -> str | None:
    text = fold(label)
    for family, patterns in QUALIFIER_FAMILIES.items():
        if any(re.search(rf"\b(?:{pattern})\b", text) for pattern in patterns):
            return family
    return None


def question_names_family_qualifier(question: object, family: str) -> bool:
    text = fold(question)
    return any(
        re.search(rf"\b(?:{pattern})\b", text)
        for pattern in QUALIFIER_FAMILIES[family]
    )


def core_tokens(label: object, family: str) -> set[str]:
    text = fold(label)
    for pattern in QUALIFIER_FAMILIES[family]:
        text = re.sub(rf"\b(?:{pattern})\b", " ", text)
    return {
        token for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 1 and token not in CORE_STOPWORDS and not token.isdigit()
    }


def same_component_metric(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    overlap = left & right
    return len(overlap) >= 2 and len(overlap) / min(len(left), len(right)) >= 0.75


def unqualified_component_candidate(
    frame: Any,
    selected_row: int,
    column: int,
    question: object,
    *,
    max_span: int = 12,
) -> dict[str, Any] | None:
    if not (0 <= selected_row < len(frame) and 0 <= column < len(frame.columns)):
        return None
    selected_label = physical_label(frame, selected_row, column)
    family = matched_family(selected_label)
    if family is None or question_names_family_qualifier(question, family):
        return None
    selected_core = core_tokens(selected_label, family)
    question_tokens = {
        token for token in re.findall(r"[a-z0-9]+", fold(question))
        if len(token) > 1 and token not in CORE_STOPWORDS and not token.isdigit()
    }
    if len(selected_core) < 2 or not selected_core <= question_tokens:
        return None
    selected_value = financial_number(frame.iloc[selected_row, column])
    if selected_value is None:
        return None

    end = min(len(frame), selected_row + max_span + 1)
    for candidate_row in range(selected_row + 1, end):
        candidate_value = financial_number(frame.iloc[candidate_row, column])
        if candidate_value is None:
            continue
        candidate_label = physical_label(frame, candidate_row, column)
        if candidate_label and not label_is_total(candidate_label):
            continue
        earliest = max(0, candidate_row - max_span)
        for start in range(earliest, selected_row + 1):
            operands: list[dict[str, Any]] = []
            valid = True
            for row in range(start, candidate_row):
                value = financial_number(frame.iloc[row, column])
                if value is None:
                    continue
                label = physical_label(frame, row, column)
                if label_is_total(label):
                    operands = []
                    continue
                if matched_family(label) != family or not same_component_metric(
                    selected_core,
                    core_tokens(label, family),
                ):
                    valid = False
                    break
                operands.append({
                    "row": row,
                    "label": label,
                    "raw": str(frame.iloc[row, column]),
                    "value": value,
                })
            if not valid or len(operands) < 2 or selected_row not in {item["row"] for item in operands}:
                continue
            if sum((item["value"] for item in operands), Decimal(0)) != candidate_value:
                continue
            return {
                "confidence": "high",
                "qualifier_family": family,
                "selected_row": selected_row,
                "selected_column": column,
                "selected_label": selected_label,
                "selected_raw": str(frame.iloc[selected_row, column]),
                "candidate_row": candidate_row,
                "candidate_label": candidate_label,
                "candidate_raw": str(frame.iloc[candidate_row, column]),
                "component_rows": [
                    {key: item[key] for key in ("row", "label", "raw")}
                    for item in operands
                ],
            }
    return None


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
        selected_locations = {
            (str(item.get("source_table")), int(item.get("row_idx", -1)), int(item.get("col_idx", -1)))
            for item in record.get("cells", [])
        }
        for cell in record.get("cells", []):
            scanned += 1
            try:
                frame = parsed_table(str(cell["source_table"]))
                detected = unqualified_component_candidate(
                    frame,
                    int(cell["row_idx"]),
                    int(cell["col_idx"]),
                    record.get("question", ""),
                )
            except Exception as error:
                errors.append({
                    "id": qid,
                    "source_table": cell.get("source_table"),
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            if detected and finding_is_covered_by_selected_cells(
                detected,
                str(cell["source_table"]),
                int(cell["col_idx"]),
                selected_locations,
            ):
                detected = None
            if detected:
                findings.append({
                    "id": qid,
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "metric_key": cell.get("metric_key"),
                    "year": cell.get("year"),
                    "source_table": cell.get("source_table"),
                    **detected,
                })

    question_ids = sorted({int(item["id"]) for item in findings})
    result = {
        "lineage": str(args.lineage),
        "physical_cells_scanned": scanned,
        "finding_count": len(findings),
        "question_count": len(question_ids),
        "question_ids": question_ids,
        "resolution_error_count": len(errors),
        "policy": "Unrequested qualifier + same-family siblings + exact additive total; verify source before repair.",
        "findings": findings,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "physical_cells_scanned", "finding_count", "question_count",
        "question_ids", "resolution_error_count",
    )}, ensure_ascii=False, indent=2))
    print("output", args.out)


if __name__ == "__main__":
    main()
