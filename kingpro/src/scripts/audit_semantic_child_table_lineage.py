"""Find operands bound to a summary table when a semantic child table is exact.

Financial notes often disclose one amount twice: first as a summary row and
then in a nearby detail table whose component labels explain the number.  A
question generated from the detail table can therefore execute correctly while
``relevant_tables`` points at the summary table, depressing table retrieval
quality.  q971 is the motivating example: ``Giá vốn dịch vụ môi giới`` equals
the following table's additive ``Cộng`` row, but only the child table names the
requested hoa-hồng components.

This audit is deliberately strict.  An alternative must be in the same report,
near the selected table, contain the exact selected raw value in the requested
year column, expose an additive total over at least two components, and add at
least two question terms absent from the current table.  Findings are review
candidates only; the script never rewrites a submission.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from audit_legacy_source_scope import fold
from audit_source_cell_semantics import header_path, parsed_table, physical_label
from audit_total_row_component_semantics import financial_number


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "build" / "catalog_enriched.jsonl"

STOPWORDS = {
    "bao", "nhieu", "cong", "ty", "tap", "doan", "ngan", "hang",
    "thuong", "mai", "co", "phan", "cua", "la", "trong", "tai",
    "vao", "va", "theo", "ghi", "nhan", "pham", "vi", "me", "hop",
    "nhat", "nam", "giai", "doan", "den", "cao", "thap", "lon",
    "nho", "muc", "gia", "tri", "bao", "cao", "tai", "chinh",
    "trieu", "nghin", "ty", "dong", "vnd", "usd", "so", "du",
}


def content_terms(text: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", fold(text))
        if len(token) >= 2 and token not in STOPWORDS and not token.isdigit()
    }


def bigrams(text: object) -> set[tuple[str, str]]:
    tokens = re.findall(r"[a-z0-9]+", fold(text))
    return set(zip(tokens, tokens[1:]))


def recovered_repeated_component_phrases(
    question: object,
    current_text: object,
    component_labels: list[object],
) -> list[str]:
    """Find question bigrams repeated across child rows but absent upstream."""

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for label in component_labels:
        for phrase in bigrams(label):
            counts[phrase] += 1
    recovered = {
        phrase
        for phrase, count in counts.items()
        if count >= 2 and phrase in bigrams(question) and phrase not in bigrams(current_text)
    }
    return [" ".join(phrase) for phrase in sorted(recovered)]


def is_total_row_label(label: object) -> bool:
    """Recognize a total row without mistaking ``cộng tác viên`` for total."""

    value = fold(label).strip(" .:-")
    return bool(re.fullmatch(r"(?:tong(?: cong)?|cong|total)", value))


def requested_year_matches(frame: Any, row: int, column: int, year: object) -> bool:
    """Reject an equal comparative value sitting under a different year."""

    target = str(year).strip()
    years = {
        match
        for value in [str(frame.columns[column]), *header_path(frame, column, row)]
        for match in re.findall(r"\b(?:19|20)\d{2}\b", str(value))
    }
    return not years or target in years


def additive_components(
    frame: Any,
    total_row: int,
    column: int,
    *,
    max_span: int = 16,
) -> list[dict[str, Any]] | None:
    """Return the nearest exact additive block ending at ``total_row``."""

    total = financial_number(frame.iloc[total_row, column])
    if total is None:
        return None
    start_limit = max(0, total_row - max_span)
    candidates: list[dict[str, Any]] = []
    for row in range(total_row - 1, start_limit - 1, -1):
        label = physical_label(frame, row, column)
        if is_total_row_label(label):
            break
        value = financial_number(frame.iloc[row, column])
        if value is None:
            continue
        candidates.append({
            "row": row,
            "label": label,
            "raw": str(frame.iloc[row, column]),
            "value": value,
        })
    candidates.reverse()
    if len(candidates) < 2:
        return None
    if sum((item["value"] for item in candidates), Decimal(0)) != total:
        return None
    return candidates


def semantic_child_candidate(
    *,
    question: object,
    year: object,
    raw: object,
    current_search_text: object,
    current_label: object,
    alternative_frame: Any,
    alternative_search_text: object,
    max_missing_terms: int = 2,
) -> dict[str, Any] | None:
    """Return the best exact child-table coordinate, if one is convincing."""

    q_terms = content_terms(question)
    current_terms = content_terms(current_search_text) | content_terms(current_label)
    target = str(raw).strip()
    best: dict[str, Any] | None = None
    for row in range(len(alternative_frame)):
        for column in range(len(alternative_frame.columns)):
            if str(alternative_frame.iloc[row, column]).strip() != target:
                continue
            if not requested_year_matches(alternative_frame, row, column, year):
                continue
            total_label = physical_label(alternative_frame, row, column)
            if total_label and not is_total_row_label(total_label):
                continue
            components = additive_components(alternative_frame, row, column)
            if not components:
                continue
            component_text = " | ".join(str(item["label"]) for item in components)
            component_terms = content_terms(component_text)
            alternative_terms = content_terms(alternative_search_text) | component_terms
            missing_terms = sorted(q_terms & alternative_terms - current_terms)
            component_overlap = sorted(q_terms & component_terms)
            if len(missing_terms) < max_missing_terms or len(component_overlap) < 2:
                continue
            repeated_phrases = recovered_repeated_component_phrases(
                question,
                f"{current_search_text} | {current_label}",
                [item["label"] for item in components],
            )
            result = {
                "candidate_row": row,
                "candidate_column": column,
                "candidate_label": total_label,
                "candidate_raw": target,
                "header_path": header_path(alternative_frame, column, row),
                "missing_question_terms_recovered": missing_terms,
                "component_question_terms": component_overlap,
                "recovered_repeated_component_phrases": repeated_phrases,
                "confidence": "high" if repeated_phrases else "review",
                "components": [
                    {key: item[key] for key in ("row", "label", "raw")}
                    for item in components
                ],
                "priority": (
                    100 * bool(repeated_phrases)
                    + 10 * len(missing_terms)
                    + len(component_overlap)
                ),
            }
            if best is None or int(result["priority"]) > int(best["priority"]):
                best = result
    return best


def load_catalog(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_ref: dict[str, dict[str, Any]] = {}
    by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ref = str(row["table_ref"])
            by_ref[ref] = row
            by_report[str(row["report_id"])].append(row)
    return by_ref, dict(by_report)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage", type=Path)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-line-distance", type=int, default=40)
    args = parser.parse_args()

    lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
    catalog, by_report = load_catalog(args.catalog)
    selected_by_question = {
        int(record["id"]): {str(cell.get("source_table", "")) for cell in record.get("cells", [])}
        for record in lineage.get("records", [])
    }
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    alternatives_checked = 0

    for record in lineage.get("records", []):
        qid = int(record["id"])
        for cell in record.get("cells", []):
            current_ref = str(cell.get("source_table", ""))
            current_meta = catalog.get(current_ref)
            if current_meta is None:
                continue
            report = str(current_meta["report_id"])
            current_line = int(current_meta.get("line", 0))
            for alternative_meta in by_report.get(report, []):
                alternative_ref = str(alternative_meta["table_ref"])
                if alternative_ref in selected_by_question[qid]:
                    continue
                distance = abs(int(alternative_meta.get("line", 0)) - current_line)
                if distance == 0 or distance > args.max_line_distance:
                    continue
                alternatives_checked += 1
                try:
                    alternative_frame = parsed_table(alternative_ref)
                    candidate = semantic_child_candidate(
                        question=record.get("question", ""),
                        year=cell.get("year", ""),
                        raw=cell.get("raw_physical", cell.get("raw_manifest", "")),
                        current_search_text=current_meta.get("search_text", ""),
                        current_label=cell.get("source_label", ""),
                        alternative_frame=alternative_frame,
                        alternative_search_text=alternative_meta.get("search_text", ""),
                    )
                except Exception as error:
                    errors.append({
                        "id": qid,
                        "current_table": current_ref,
                        "alternative_table": alternative_ref,
                        "error": f"{type(error).__name__}: {error}",
                    })
                    continue
                if candidate:
                    findings.append({
                        "id": qid,
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "year": cell.get("year"),
                        "metric_key": cell.get("metric_key"),
                        "current_table": current_ref,
                        "current_label": cell.get("source_label"),
                        "alternative_table": alternative_ref,
                        "line_distance": distance,
                        "alternative_search_text": alternative_meta.get("search_text", ""),
                        **candidate,
                    })

    findings.sort(key=lambda item: (-int(item["priority"]), int(item["id"]), str(item["year"])))
    question_ids = sorted({int(item["id"]) for item in findings})
    high_confidence_ids = sorted({
        int(item["id"]) for item in findings if item.get("confidence") == "high"
    })
    payload = {
        "kind": "semantic_child_table_lineage_review",
        "lineage": str(args.lineage),
        "alternatives_checked": alternatives_checked,
        "finding_count": len(findings),
        "question_count": len(question_ids),
        "question_ids": question_ids,
        "high_confidence_question_ids": high_confidence_ids,
        "resolution_error_count": len(errors),
        "policy": (
            "Same report + nearby table + exact raw/year + additive total + at least two "
            "question terms recovered. Review before replacing or adding relevant_tables."
        ),
        "findings": findings,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "alternatives_checked", "finding_count", "question_count",
            "question_ids", "high_confidence_question_ids", "resolution_error_count",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
