"""Find unqualified totals that read one component instead of an additive row total.

Financial notes frequently omit the text ``Total``/``Cộng`` on their final
row.  A lexical retriever can therefore bind an unqualified question such as
``tổng dự phòng rủi ro`` to a named component (for example ``tại Việt Nam``)
and miss the following blank row that equals the sum of all components.

This audit consumes the physical cell-lineage report.  It emits a narrow,
source-verifiable pattern only when:

* the question explicitly asks for a total;
* the selected row is not already labelled as a total; and
* a later blank/total row in the same physical column exactly equals at least
  two contiguous component values, including the selected value.

The tool is read-only.  Findings are candidate repairs, never automatic
submission changes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from audit_legacy_source_scope import fold
from audit_source_cell_semantics import parsed_table, physical_label


TOTAL_LABEL_RE = re.compile(r"\b(tong|tong cong|cong|cong cong)\b")
TOTAL_START_RE = re.compile(r"\btong(?:\s+cong)?\b")
TOTAL_TARGET_STOP_RE = re.compile(
    r"\b(tai|vao|giua|so voi|la bao nhieu|la may|cao nhat|lon nhat|"
    r"trong nam|tai nam|nam co)\b"
)
ENTITY_OWNER_RE = re.compile(
    r"\bcua\s+(?=(?:cong ty|ctcp|tap doan|tong cong ty|ngan hang|"
    r"tong cong ty|doanh nghiep|tap doan))"
)
TOKEN_STOPWORDS = {
    "tong", "cong", "so", "du", "cuoi", "dau", "nam", "ky", "gia",
    "tri", "la", "bao", "nhieu", "may", "dong", "trieu", "ty", "va",
    "cac", "khoan", "muc", "tai", "ngay", "thuyet", "minh",
}


def financial_number(value: object) -> Decimal | None:
    """Parse a displayed accounting scalar for within-column additivity checks."""

    text = re.sub(r"\s+", "", str(value)).replace("−", "-")
    if not text or text.casefold() == "nan" or text in {"-", "--"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").rstrip("%")
    if not re.fullmatch(r"-?\d+(?:[.,]\d+)*", text):
        return None
    # Exact equality is only compared inside one table column.  Removing both
    # separators is therefore stable for Vietnamese accounting formatting.
    compact = text.replace(".", "").replace(",", "")
    try:
        result = Decimal(compact)
    except InvalidOperation:
        return None
    return -abs(result) if negative else result


def semantic_tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", fold(value))
        if len(token) > 1 and token not in TOKEN_STOPWORDS and not token.isdigit()
    }


def total_targets(question: object) -> list[str]:
    """Extract semantic noun phrases governed by ``tổng``.

    ``Tổng Công ty`` is an entity title, not a request for an additive value.
    Stopping at ``của`` keeps later selector clauses from contaminating the
    target phrase.
    """

    text = fold(question)
    targets: list[str] = []
    for match in TOTAL_START_RE.finditer(text):
        tail = text[match.end():].strip()
        matched = match.group(0)
        if re.match(r"cong ty\b", tail) or (matched == "tong cong" and re.match(r"ty\b", tail)):
            continue
        stops = [item for item in (
            TOTAL_TARGET_STOP_RE.search(tail),
            ENTITY_OWNER_RE.search(tail),
        ) if item]
        stop_at = min((item.start() for item in stops), default=len(tail))
        target = tail[:stop_at].strip(" ,.;:-")
        if semantic_tokens(target):
            targets.append(target)
    return targets


def question_requests_total(question: object) -> bool:
    return bool(total_targets(question))


def label_is_total(label: object) -> bool:
    return bool(TOTAL_LABEL_RE.search(fold(label)))


def label_matches_total_target(question: object, label: object) -> bool:
    """Whether a selected component plausibly belongs to the requested total."""

    label_tokens = semantic_tokens(label)
    if len(label_tokens) < 2:
        return False
    for target in total_targets(question):
        target_tokens = semantic_tokens(target)
        overlap = label_tokens & target_tokens
        if len(overlap) >= 2 and len(overlap) / len(label_tokens) >= 0.5:
            return True
    return False


def label_is_explicitly_requested_component(question: object, label: object) -> bool:
    """Suppress totals when the question deliberately names the selected row."""

    label_tokens = semantic_tokens(label)
    if len(label_tokens) < 2:
        return False
    return any(label_tokens <= semantic_tokens(target) for target in total_targets(question))


def every_component_matches_total_target(
    question: object,
    components: list[dict[str, Any]],
) -> bool:
    """Reject a broader table total containing unrelated accounting metrics."""

    return bool(components) and all(
        item.get("label") and label_matches_total_target(question, item["label"])
        for item in components
    )


def finding_is_covered_by_selected_cells(
    finding: dict[str, Any],
    source_table: str,
    column: int,
    selected_locations: set[tuple[str, int, int]],
) -> bool:
    """True when the program already reads the total or every component."""

    candidate = (source_table, int(finding["candidate_row"]), column)
    if candidate in selected_locations:
        return True
    components = {
        (source_table, int(item["row"]), column)
        for item in finding.get("component_rows", [])
    }
    return bool(components) and components <= selected_locations


def additive_total_candidate(
    frame: Any,
    selected_row: int,
    column: int,
    question: object,
    *,
    max_span: int = 12,
) -> dict[str, Any] | None:
    """Return the nearest exact additive total containing ``selected_row``."""

    if not question_requests_total(question):
        return None
    if not (0 <= selected_row < len(frame) and 0 <= column < len(frame.columns)):
        return None

    selected_label = physical_label(frame, selected_row, column)
    if label_is_total(selected_label):
        return None
    if not label_matches_total_target(question, selected_label):
        return None
    if label_is_explicitly_requested_component(question, selected_label):
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
        # Totals are commonly either explicit or the final blank row.  Do not
        # treat an arbitrary named component as a possible aggregate.
        if candidate_label and not label_is_total(candidate_label):
            continue

        earliest = max(0, candidate_row - max_span)
        for start in range(earliest, selected_row + 1):
            operands: list[dict[str, Any]] = []
            for row in range(start, candidate_row):
                value = financial_number(frame.iloc[row, column])
                if value is None:
                    continue
                label = physical_label(frame, row, column)
                # An existing subtotal starts a new additive block; including
                # it with its children would double count.
                if label_is_total(label):
                    operands = []
                    continue
                operands.append({
                    "row": row,
                    "label": label,
                    "raw": str(frame.iloc[row, column]),
                    "value": value,
                })
            operand_rows = {int(item["row"]) for item in operands}
            if selected_row not in operand_rows or len(operands) < 2:
                continue
            if sum((item["value"] for item in operands), Decimal(0)) != candidate_value:
                continue
            if not every_component_matches_total_target(question, operands):
                continue
            return {
                "confidence": "high",
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
        question = record.get("question", "")
        if not question_requests_total(question):
            continue
        qid = int(record["id"])
        selected_locations = {
            (str(item.get("source_table")), int(item.get("row_idx", -1)), int(item.get("col_idx", -1)))
            for item in record.get("cells", [])
        }
        for cell in record.get("cells", []):
            scanned += 1
            try:
                frame = parsed_table(str(cell["source_table"]))
                detected = additive_total_candidate(
                    frame,
                    int(cell["row_idx"]),
                    int(cell["col_idx"]),
                    question,
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
                    "question": question,
                    "answer": record.get("answer"),
                    "metric_key": cell.get("metric_key"),
                    "year": cell.get("year"),
                    "source_table": cell.get("source_table"),
                    **detected,
                })

    question_ids = sorted({int(item["id"]) for item in findings})
    result = {
        "lineage": str(args.lineage),
        "total_intent_cells_scanned": scanned,
        "finding_count": len(findings),
        "question_count": len(question_ids),
        "question_ids": question_ids,
        "resolution_error_count": len(errors),
        "policy": "Exact additive blank/total-row candidates only; verify full source and recompute before repair.",
        "findings": findings,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "total_intent_cells_scanned",
        "finding_count",
        "question_count",
        "question_ids",
        "resolution_error_count",
    )}, ensure_ascii=False, indent=2))
    print("output", args.out)


if __name__ == "__main__":
    main()
