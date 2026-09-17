"""Audit live retrieval against source-proven table bindings.

This is deliberately *not* a hidden-gold evaluator.  It measures whether the
product retriever can recover the tables that an already source-audited
candidate actually executes.  The distinction matters: the report is useful
for regression testing and failure analysis, but must never be presented as a
BTC leaderboard score.

Two retrieval paths are reported:

* ``pipeline`` uses documents found by ``retrieve_decomposed`` and therefore
  measures the complete document -> table path used by the product.
* ``oracle_documents`` supplies only the candidate's declared source documents
  to ``tables_in_reports``.  It isolates ranking *inside* a correct report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.retrieval.bm25_index import (  # noqa: E402
    extract_all_facets,
    retrieve_decomposed,
    tables_in_reports,
)
from kingpro.retrieval.table_reranker import rerank_tables  # noqa: E402


DEFAULT_CANDIDATE = ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
DEFAULT_OUTPUT = ROOT / "build" / "demo_compliance" / "retrieval_provenance_recovery_v1.json"
KS = (1, 3, 5, 8, 12, 20, 40)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _report_id(table_ref: str) -> str:
    return str(table_ref).split("|", 1)[0]


def _unique(values):
    return list(dict.fromkeys(str(value) for value in values if value))


def _target_tables(row: dict, audit: dict | None) -> list[str]:
    """Prefer executed source bindings; fall back to declared retrieval labels."""
    if audit:
        refs = _unique(
            source.get("table_ref")
            for source in audit.get("sources", [])
            if not str(source.get("metric", "")).startswith("recall:")
        )
        if refs:
            return refs
    return _unique(row.get("relevant_tables", []))


def _set_recall(targets: set[str], predictions: list[str]) -> float:
    if not targets:
        return 0.0
    return len(targets.intersection(predictions)) / len(targets)


def _reciprocal_rank(targets: set[str], predictions: list[str]) -> float:
    for index, prediction in enumerate(predictions, 1):
        if prediction in targets:
            return 1.0 / index
    return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bucket(row: dict, facets: dict, target_count: int) -> list[str]:
    question = str(row.get("question", "")).casefold()
    buckets = ["all"]
    buckets.append("multi_table" if target_count > 1 else "single_table")
    buckets.append("analytic" if facets.get("analytic") else "lookup")
    buckets.append("multi_entity_year" if len(facets.get("tickers", [])) * len(facets.get("years", [])) > 1 else "single_entity_year")
    if any(token in question for token in ("ngân hàng", "cho vay", "tiền gửi", "tín dụng")):
        buckets.append("banking_language")
    if any(token in question for token in ("tỷ lệ", "biên", "roe", "roa", "%")):
        buckets.append("ratio_language")
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Diagnostic prefix only; 0 audits all rows.")
    parser.add_argument(
        "--label-weight",
        type=float,
        default=0.0,
        help="Apply source-grounded CSV label reranking with this weight; 0 preserves BM25.",
    )
    parser.add_argument(
        "--facet-backfill",
        action="store_true",
        help="Backfill a preferred ticker/year/scope only when one catalog report matches.",
    )
    parser.add_argument(
        "--scope-neutral-backfill",
        action="store_true",
        help="Prefer a scope-neutral report only when ticker/year maps to one unique catalog report.",
    )
    parser.add_argument(
        "--opening-year-backfill",
        action="store_true",
        help="Add prior reports only for explicit opening-period semantics.",
    )
    parser.add_argument(
        "--formula-year-backfill",
        action="store_true",
        help="Add one prior report for metrics defined on an average balance.",
    )
    parser.add_argument(
        "--growth-scan-year-backfill",
        action="store_true",
        help="Add the baseline report for extreme revenue-growth scans over a period.",
    )
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    submission = _read_json(candidate / "submission.json")
    if args.limit:
        submission = submission[: args.limit]
    audit_path = candidate / "source_audit.json"
    source_audit = {}
    if audit_path.is_file():
        source_audit = {int(row["id"]): row for row in _read_json(audit_path)}
    catalog = {}
    if args.label_weight:
        with (ROOT / "build" / "catalog.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                catalog_row = json.loads(line)
                catalog[str(catalog_row["table_ref"])] = catalog_row

    # Warm the mmap/cache before timed iteration.
    retrieve_decomposed("warm")

    aggregate = defaultdict(lambda: defaultdict(list))
    misses = []
    document_misses = []
    target_histogram = Counter()
    for position, row in enumerate(submission, 1):
        question_id = int(row["id"])
        question = str(row["question"])
        targets = set(_target_tables(row, source_audit.get(question_id)))
        if not targets:
            continue
        target_docs = set(_unique(row.get("relevant_docs", []))) or {_report_id(ref) for ref in targets}
        target_histogram[len(targets)] += 1
        facets = extract_all_facets(question)

        document_hits = retrieve_decomposed(
            question,
            per_pair=1,
            cap=40,
            backfill_unique_preferred_scope=args.facet_backfill,
            backfill_unique_scope_neutral=args.scope_neutral_backfill,
            backfill_explicit_opening_year=args.opening_year_backfill,
            backfill_financial_formula_year=args.formula_year_backfill,
            backfill_growth_scan_year=args.growth_scan_year_backfill,
        )
        predicted_docs = _unique(_report_id(hit["table_ref"]) for hit in document_hits)
        pipeline_hits = tables_in_reports(question, predicted_docs, n=max(KS))
        oracle_hits = tables_in_reports(question, sorted(target_docs), n=max(KS))
        if args.label_weight:
            pipeline_hits = rerank_tables(
                question,
                pipeline_hits,
                catalog,
                ROOT / "build" / "tables",
                label_weight=args.label_weight,
            )
            oracle_hits = rerank_tables(
                question,
                oracle_hits,
                catalog,
                ROOT / "build" / "tables",
                label_weight=args.label_weight,
            )
        pipeline_refs = _unique(hit["table_ref"] for hit in pipeline_hits)
        oracle_refs = _unique(hit["table_ref"] for hit in oracle_hits)

        document_recall = _set_recall(target_docs, predicted_docs)
        if document_recall < 1.0:
            document_misses.append(
                {
                    "id": question_id,
                    "question": question,
                    "target_docs": sorted(target_docs),
                    "predicted_docs": predicted_docs[:8],
                    "recall": round(document_recall, 6),
                }
            )

        record = {
            "id": question_id,
            "question": question,
            "target_tables": sorted(targets),
            "target_docs": sorted(target_docs),
            "facets": facets,
            "document_recall": document_recall,
            "pipeline_rank": next((i for i, ref in enumerate(pipeline_refs, 1) if ref in targets), None),
            "oracle_rank": next((i for i, ref in enumerate(oracle_refs, 1) if ref in targets), None),
            "pipeline_top": pipeline_refs[:8],
            "oracle_top": oracle_refs[:8],
        }
        record["pipeline_recall_at_8"] = _set_recall(targets, pipeline_refs[:8])
        record["oracle_recall_at_8"] = _set_recall(targets, oracle_refs[:8])
        if record["oracle_recall_at_8"] < 1.0:
            misses.append(record)

        for bucket in _bucket(row, facets, len(targets)):
            aggregate[bucket]["document_recall"].append(document_recall)
            aggregate[bucket]["pipeline_mrr40"].append(_reciprocal_rank(targets, pipeline_refs[:40]))
            aggregate[bucket]["oracle_mrr40"].append(_reciprocal_rank(targets, oracle_refs[:40]))
            for k in KS:
                aggregate[bucket][f"pipeline_recall_at_{k}"].append(_set_recall(targets, pipeline_refs[:k]))
                aggregate[bucket][f"oracle_recall_at_{k}"].append(_set_recall(targets, oracle_refs[:k]))

        if position % 100 == 0:
            print(f"audited {position}/{len(submission)}", flush=True)

    summaries = {}
    for bucket, metrics in sorted(aggregate.items()):
        summaries[bucket] = {
            "questions": len(metrics["document_recall"]),
            **{name: round(_mean(values), 6) for name, values in sorted(metrics.items())},
        }

    misses.sort(
        key=lambda row: (
            row["oracle_recall_at_8"],
            math.inf if row["oracle_rank"] is None else row["oracle_rank"],
            row["id"],
        )
    )
    payload = {
        "kind": "source_proven_retrieval_regression_not_btc_gold",
        "candidate": candidate.name,
        "label_weight": args.label_weight,
        "facet_backfill": args.facet_backfill,
        "scope_neutral_backfill": args.scope_neutral_backfill,
        "opening_year_backfill": args.opening_year_backfill,
        "formula_year_backfill": args.formula_year_backfill,
        "growth_scan_year_backfill": args.growth_scan_year_backfill,
        "audited_questions": summaries.get("all", {}).get("questions", 0),
        "ks": list(KS),
        "target_table_count_histogram": {str(key): value for key, value in sorted(target_histogram.items())},
        "summary": summaries,
        "document_miss_count": len(document_misses),
        "oracle_table_miss_at_8_count": len(misses),
        "document_misses": document_misses[:100],
        "oracle_table_misses_at_8": misses[:200],
        "claim_limits": [
            "Targets come from source-audited executable bindings, not the hidden BTC gold labels.",
            "Metrics are valid for product regression and error topology only.",
            "They must not be reported as leaderboard F2, precision, recall, or expected hidden score.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "audited_questions": payload["audited_questions"],
        "all": summaries.get("all", {}),
        "document_miss_count": len(document_misses),
        "oracle_table_miss_at_8_count": len(misses),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
