"""Summarize source-bound retrieval misses without using question IDs as features.

The coverage report is produced by ``audit_document_retrieval_coverage.py``.
This helper joins its misses to public catalog metadata and the candidate's
source bindings so repeated structural causes can be inspected reproducibly.
It never changes a submission and never treats the bindings as BTC hidden gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.retrieval.bm25_index import (  # noqa: E402
    extract_all_facets,
    series_report_variant_limit,
)
DEFAULT_COVERAGE = (
    ROOT / "build" / "demo_compliance" / "document_retrieval_coverage_semantic_v22_final.json"
)
DEFAULT_CANDIDATE = ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
DEFAULT_OUTPUT = ROOT / "build" / "demo_compliance" / "retrieval_v22_miss_patterns.json"


def _report_meta() -> dict[str, dict[str, str]]:
    reports: dict[str, dict[str, str]] = {}
    with (ROOT / "build" / "catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            report_id = str(row["report_id"])
            reports.setdefault(
                report_id,
                {
                    "ticker": str(row.get("ticker", "")),
                    "year": str(row.get("year", "")),
                    "scope": str(row.get("scope", "")),
                },
            )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--mode", default="1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    rows = json.loads((args.candidate / "submission.json").read_text(encoding="utf-8"))
    rows_by_id = {int(row["id"]): row for row in rows}
    reports = _report_meta()
    report_groups: dict[tuple[str, str, str], list[str]] = {}
    for report_id, metadata in reports.items():
        key = (metadata["ticker"], metadata["year"], metadata["scope"])
        report_groups.setdefault(key, []).append(report_id)
    for report_ids in report_groups.values():
        report_ids.sort()
    misses = coverage["modes"][str(args.mode)]["misses"]

    reason_counts: Counter[str] = Counter()
    normalized = []
    for miss in misses:
        question_id = int(miss["id"])
        row = rows_by_id[question_id]
        missing_docs = []
        for item in miss["missing"]:
            reason = str(item["reason"])
            reason_counts[reason] += 1
            missing_docs.append(
                {
                    **item,
                    "metadata": reports.get(str(item["document"]), {}),
                }
            )
        normalized.append(
            {
                "id": question_id,
                "question": str(row["question"]),
                "facets": miss["facets"],
                "target_count": len(miss["targets"]),
                "prediction_count": len(miss["predictions"]),
                "targets": [
                    {"report_id": report_id, **reports.get(report_id, {})}
                    for report_id in miss["targets"]
                ],
                "predictions": [
                    {"report_id": report_id, **reports.get(report_id, {})}
                    for report_id in miss["predictions"]
                ],
                "missing": missing_docs,
            }
        )

    no_year_entity_questions = []
    ambiguous_series_questions = []
    for row in rows:
        facets = extract_all_facets(str(row["question"]))
        if not facets["tickers"] or facets["years"]:
            continue
        target_meta = [reports.get(str(report_id), {}) for report_id in row.get("relevant_docs", [])]
        no_year_entity_questions.append(
            {
                "id": int(row["id"]),
                "question": str(row["question"]),
                "tickers": facets["tickers"],
                "target_years": sorted(
                    {str(meta.get("year", "")) for meta in target_meta if meta.get("year")}
                ),
                "target_scopes": sorted(
                    {str(meta.get("scope", "")) for meta in target_meta if meta.get("scope")}
                ),
                "target_count": len(row.get("relevant_docs", [])),
            }
        )
    for row in rows:
        facets = extract_all_facets(str(row["question"]))
        if series_report_variant_limit(
            facets,
            1,
            question=str(row["question"]),
            retain_ambiguous_series_reports=True,
        ) == 1:
            continue
        ambiguous_groups = []
        for ticker in facets["tickers"]:
            for year in facets["years"]:
                key = (ticker, year, facets["scope"])
                variants = report_groups.get(key, [])
                if len(variants) < 2:
                    continue
                ambiguous_groups.append(
                    {
                        "ticker": ticker,
                        "year": year,
                        "scope": facets["scope"],
                        "variants": variants,
                        "target_variants": [
                            report_id
                            for report_id in variants
                            if report_id in set(row.get("relevant_docs", []))
                        ],
                    }
                )
        if ambiguous_groups:
            ambiguous_series_questions.append(
                {
                    "id": int(row["id"]),
                    "question": str(row["question"]),
                    "groups": ambiguous_groups,
                }
            )

    payload = {
        "kind": "source_bound_retrieval_miss_pattern_audit_not_btc_gold",
        "coverage": str(args.coverage),
        "candidate": args.candidate.name,
        "mode": str(args.mode),
        "question_count": len(rows),
        "missed_question_count": len(normalized),
        "missing_document_count": sum(reason_counts.values()),
        "reason_counts": dict(sorted(reason_counts.items())),
        "no_explicit_year_with_entity_count": len(no_year_entity_questions),
        "no_explicit_year_with_entities": no_year_entity_questions,
        "ambiguous_series_question_count": len(ambiguous_series_questions),
        "ambiguous_series_questions": ambiguous_series_questions,
        "misses": normalized,
        "claim_limits": [
            "Targets are executable source bindings, not BTC hidden-gold labels.",
            "Question IDs are reported for audit only and must never become retrieval features.",
            "A rule is promotable only after a full-corpus precision/recall/F2 regression.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "missed_questions": len(normalized),
                "missing_documents": sum(reason_counts.values()),
                "reason_counts": dict(sorted(reason_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
