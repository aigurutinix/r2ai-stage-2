"""Find exact-value tables whose row semantics fit the question better.

One financial amount is often repeated in a statement, a summary note and a
detail note.  Execution can therefore be numerically correct while
``relevant_tables`` names the wrong disclosure.  This read-only audit searches
nearby tables in the same report for the exact physical value and requested
year, then compares question/row-label phrase overlap.  It never rewrites a
submission; every finding remains a source-review candidate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_legacy_source_scope import fold
from audit_semantic_child_table_lineage import (
    STOPWORDS,
    load_catalog,
    requested_year_matches,
)
from audit_source_cell_semantics import header_path, parsed_table, physical_label


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "build" / "catalog_enriched.jsonl"


def semantic_tokens(text: object) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", fold(text))
        if len(token) >= 2 and token not in STOPWORDS and not token.isdigit()
    ]


def semantic_bigrams(text: object) -> set[tuple[str, str]]:
    tokens = semantic_tokens(text)
    return set(zip(tokens, tokens[1:]))


def expected_statement_family(metric_key: object) -> str:
    """Return the canonical primary-statement family encoded by a metric key."""

    prefix = str(metric_key or "").split(":", 1)[0].lower()
    return {
        "cdkt": "balance_sheet",
        "kqkd": "income_statement",
        "lctt": "cash_flow",
    }.get(prefix, "")


def statement_family(text: object) -> str:
    """Classify a table only when its own text contains a strong family cue."""

    value = fold(text)
    if (
        "luu chuyen tien" in value
        or "lu chuyen tien" in value
        or re.search(r"\blu\W*u\s+chuyen tien", value)
    ):
        return "cash_flow"
    if "ket qua hoat dong kinh doanh" in value:
        return "income_statement"
    if "bang can doi ke toan" in value or "bao cao tinh hinh tai chinh" in value:
        return "balance_sheet"
    return ""


def hard_semantic_conflict(
    *,
    question: object,
    metric_key: object,
    current_label: object,
    candidate_label: object,
) -> str:
    """Name a contradiction that lexical overlap must never override."""

    query = fold(question)
    current = fold(current_label)
    candidate = fold(candidate_label)
    metric = str(metric_key or "").lower()

    # Exact values frequently repeat between ending and weighted-average share
    # disclosures.  They are different concepts even when the number happens
    # to be equal in a no-issuance year.
    ending_cue = "cuoi" in query or "so du" in query or "ending" in metric
    if ending_cue and "binh quan" in candidate:
        return "ending-vs-average"
    if "shares_outstanding" in metric or "co phieu" in query:
        if ("duoc duyet" in candidate or "da phat hanh" in candidate) and "luu hanh" not in candidate:
            return "outstanding-vs-authorized-issued"

    # A question/current row that explicitly asks for a total must not be
    # replaced by one component merely because the component label has more
    # content words.
    total_markers = ("tong", "cong")
    if "tong" in query and any(marker in current for marker in total_markers):
        if not any(marker in candidate for marker in total_markers):
            return "total-vs-component"
    return ""


def coordinate_semantics(
    *,
    question: object,
    raw: object,
    year: object,
    current_text: object,
    current_label: object,
    alternative_frame: Any,
    alternative_text: object,
    metric_key: object = "",
    current_table_cell_count: int = 1,
) -> list[dict[str, Any]]:
    """Return exact coordinates with a measurable semantic gain."""

    target = str(raw).strip()
    # Empty/placeholder cells repeat throughout financial tables and carry no
    # identity.  Treating '-' as an exact value creates arbitrary alternatives.
    if target in {"", "-", "–", "—", "nan", "None"}:
        return []
    expected_family = expected_statement_family(metric_key)
    alternative_family = statement_family(alternative_text)
    if expected_family and alternative_family and alternative_family != expected_family:
        return []
    question_terms = set(semantic_tokens(question))
    question_bigrams = semantic_bigrams(question)
    current_label_terms = set(semantic_tokens(current_label))
    current_terms = set(semantic_tokens(f"{current_label} | {current_text}"))
    current_bigrams = semantic_bigrams(f"{current_label} | {current_text}")
    findings: list[dict[str, Any]] = []

    for row in range(len(alternative_frame)):
        for column in range(len(alternative_frame.columns)):
            if str(alternative_frame.iloc[row, column]).strip() != target:
                continue
            if not requested_year_matches(alternative_frame, row, column, year):
                continue
            label = physical_label(alternative_frame, row, column)
            candidate_header_path = header_path(alternative_frame, column, row)
            coordinate_family = statement_family(
                f"{alternative_text} | {label} | {' | '.join(candidate_header_path)}"
            )
            if expected_family and coordinate_family and coordinate_family != expected_family:
                continue
            conflict = hard_semantic_conflict(
                question=question,
                metric_key=metric_key,
                current_label=current_label,
                candidate_label=label,
            )
            if conflict:
                continue
            label_terms = set(semantic_tokens(label))
            combined = f"{label} | {alternative_text}"
            alternative_terms = set(semantic_tokens(combined))
            alternative_bigrams = semantic_bigrams(combined)
            recovered_terms = sorted(question_terms & (alternative_terms - current_terms))
            recovered_label_bigrams = sorted(
                question_bigrams & semantic_bigrams(label) - current_bigrams
            )
            recovered_table_bigrams = sorted(
                question_bigrams & alternative_bigrams - current_bigrams
            )
            current_label_overlap = sorted(question_terms & current_label_terms)
            alternative_label_overlap = sorted(question_terms & label_terms)
            label_gain = len(alternative_label_overlap) - len(current_label_overlap)

            # A high-confidence row must itself name a missing question phrase;
            # broad table context alone is only a review hint.
            high = bool(recovered_label_bigrams) and len(alternative_label_overlap) >= 2 and label_gain >= 1
            review = (
                len(recovered_terms) >= 2
                and bool(recovered_table_bigrams)
                and len(alternative_label_overlap) >= 2
                and len(alternative_label_overlap) >= len(current_label_overlap)
            )
            context_rule = ""
            # If the current table already supplies multiple operands for the
            # same question, co-location is positive lineage evidence.  A
            # repeated value elsewhere can remain a review hint but cannot by
            # itself justify moving one operand and fragmenting the proof.
            if high and current_table_cell_count > 1:
                high = False
                review = True
                context_rule = "preserve-co-located-operands"
            if not high and not review:
                continue
            findings.append(
                {
                    "candidate_row": row,
                    "candidate_column": column,
                    "candidate_label": label,
                    "candidate_raw": target,
                    "header_path": candidate_header_path,
                    "current_label_overlap": current_label_overlap,
                    "candidate_label_overlap": alternative_label_overlap,
                    "recovered_question_terms": recovered_terms,
                    "recovered_label_bigrams": [" ".join(value) for value in recovered_label_bigrams],
                    "recovered_table_bigrams": [" ".join(value) for value in recovered_table_bigrams],
                    "confidence": "high" if high else "review",
                    "priority": 100 * high + 10 * len(recovered_label_bigrams) + 3 * label_gain + len(recovered_terms),
                    "context_rule": context_rule,
                    "current_table_cell_count": current_table_cell_count,
                }
            )
    return findings


def audit(lineage_path: Path, catalog_path: Path, max_line_distance: int) -> dict[str, Any]:
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    catalog, by_report = load_catalog(catalog_path)
    selected_by_question = {
        int(record["id"]): {str(cell.get("source_table", "")) for cell in record.get("cells", [])}
        for record in lineage.get("records", [])
    }
    current_table_cell_counts: dict[tuple[int, str], int] = defaultdict(int)
    for record in lineage.get("records", []):
        qid = int(record["id"])
        for cell in record.get("cells", []):
            current_table_cell_counts[(qid, str(cell.get("source_table", "")))] += 1
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    alternatives_checked = 0

    for record in lineage.get("records", []):
        qid = int(record["id"])
        for cell_index, cell in enumerate(record.get("cells", [])):
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
                if distance == 0 or distance > max_line_distance:
                    continue
                alternatives_checked += 1
                try:
                    candidates = coordinate_semantics(
                        question=record.get("question", ""),
                        raw=cell.get("raw_physical", cell.get("raw_manifest", "")),
                        year=cell.get("year", ""),
                        current_text=current_meta.get("search_text", ""),
                        current_label=cell.get("source_label", ""),
                        alternative_frame=parsed_table(alternative_ref),
                        alternative_text=alternative_meta.get("search_text", ""),
                        metric_key=cell.get("metric_key", ""),
                        current_table_cell_count=current_table_cell_counts[(qid, current_ref)],
                    )
                except Exception as error:
                    errors.append(
                        {
                            "id": qid,
                            "current_table": current_ref,
                            "alternative_table": alternative_ref,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                for candidate in candidates:
                    findings.append(
                        {
                            "id": qid,
                            "cell_index": cell_index,
                            "question": record.get("question"),
                            "answer": record.get("answer"),
                            "year": cell.get("year"),
                            "metric_key": cell.get("metric_key"),
                            "current_table": current_ref,
                            "current_label": cell.get("source_label"),
                            "alternative_table": alternative_ref,
                            "line_distance": distance,
                            **candidate,
                        }
                    )

    findings.sort(key=lambda item: (-int(item["priority"]), int(item["id"]), int(item["cell_index"])))
    high_ids = sorted({int(item["id"]) for item in findings if item["confidence"] == "high"})
    question_ids = sorted({int(item["id"]) for item in findings})
    return {
        "kind": "exact_value_semantic_alternative_review",
        "lineage": str(lineage_path),
        "alternatives_checked": alternatives_checked,
        "finding_count": len(findings),
        "question_count": len(question_ids),
        "question_ids": question_ids,
        "high_confidence_question_ids": high_ids,
        "resolution_error_count": len(errors),
        "policy": "Same report, nearby exact raw value and requested year; candidate row/table must recover question semantics absent from current source. Review before repair.",
        "findings": findings,
        "errors": errors,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage", type=Path)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--max-line-distance", type=int, default=160)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.lineage, args.catalog, args.max_line_distance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "alternatives_checked", "finding_count", "question_count", "question_ids",
        "high_confidence_question_ids", "resolution_error_count",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
