"""Audit source-proven document coverage and diagnose missed reports.

Targets are the candidate's executable ``relevant_docs`` bindings, not BTC
hidden gold.  The audit is intended to tune the live product retriever without
using answers, question IDs as features, or leaderboard feedback.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.retrieval.bm25_index import extract_all_facets, retrieve_decomposed  # noqa: E402


DEFAULT_CANDIDATE = ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
DEFAULT_OUTPUT = ROOT / "build" / "demo_compliance" / "document_retrieval_coverage_v1.json"


def _unique(values):
    return list(dict.fromkeys(str(value) for value in values if value))


def _f2(precision: float, recall: float) -> float:
    denominator = 4.0 * precision + recall
    return 5.0 * precision * recall / denominator if denominator else 0.0


def _load_report_meta() -> tuple[dict[str, dict], dict[tuple[str, str, str], list[str]]]:
    reports: dict[str, dict] = {}
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    with (ROOT / "build" / "catalog.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            report_id = str(row["report_id"])
            if report_id in reports:
                continue
            meta = {
                "report_id": report_id,
                "ticker": str(row.get("ticker", "")),
                "year": str(row.get("year", "")),
                "scope": str(row.get("scope", "")),
            }
            reports[report_id] = meta
            groups[(meta["ticker"], meta["year"], meta["scope"])].append(report_id)
    for report_ids in groups.values():
        report_ids.sort()
    return reports, groups


def _diagnose(
    target_doc: str,
    predicted_docs: list[str],
    facets: dict,
    reports: dict[str, dict],
    groups: dict[tuple[str, str, str], list[str]],
) -> str:
    meta = reports.get(target_doc)
    if not meta:
        return "target_missing_from_catalog"
    if meta["ticker"] not in set(facets.get("tickers", [])):
        return "ticker_not_extracted"
    if meta["year"] not in set(facets.get("years", [])):
        return "year_not_extracted"
    key = (meta["ticker"], meta["year"], meta["scope"])
    same_group = set(groups.get(key, []))
    if same_group.intersection(predicted_docs):
        return "competing_report_same_entity_year_scope"
    opposite_scope = {
        report_id
        for report_id, candidate in reports.items()
        if candidate["ticker"] == meta["ticker"]
        and candidate["year"] == meta["year"]
        and candidate["scope"] != meta["scope"]
    }
    if opposite_scope.intersection(predicted_docs):
        return "scope_preference_selected_other_scope"
    return "retrieval_pool_or_ranking_miss"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-pair", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--cap", type=int, default=40)
    parser.add_argument("--pool", type=int, default=1000)
    parser.add_argument(
        "--facet-backfill",
        action="store_true",
        help="Backfill a preferred ticker/year/scope only when the catalog has one unique report.",
    )
    parser.add_argument(
        "--scope-neutral-backfill",
        action="store_true",
        help="Prefer a scope-neutral report only when ticker/year maps to one unique catalog report.",
    )
    parser.add_argument(
        "--opening-year-backfill",
        action="store_true",
        help="Add one prior report year only for explicit opening-period language.",
    )
    parser.add_argument(
        "--formula-year-backfill",
        action="store_true",
        help="Add one prior report year for metrics defined on an average balance.",
    )
    parser.add_argument(
        "--growth-scan-year-backfill",
        action="store_true",
        help="Add the baseline year for an extreme revenue-growth scan over a period.",
    )
    parser.add_argument(
        "--comparative-selection-year-backfill",
        action="store_true",
        help="Add one baseline year for bounded multi-entity comparative selection.",
    )
    parser.add_argument(
        "--comparative-series-year-backfill",
        action="store_true",
        help="Add one source-dependency year for bounded multi-entity comparative series.",
    )
    parser.add_argument(
        "--sparse-series-year-backfill",
        action="store_true",
        help="Fill internal predecessor dependencies for sparse analytic single-entity series.",
    )
    parser.add_argument(
        "--ambiguous-series-reports",
        action="store_true",
        help="Retain at most two physical filings for ambiguous long single-entity series facets.",
    )
    parser.add_argument(
        "--implicit-scope-minimum-ratio",
        type=float,
        default=0.0,
        help="Retain an opposite-scope filing only when its BM25 score exceeds this ratio.",
    )
    parser.add_argument(
        "--implicit-ownership-scope-minimum-ratio",
        type=float,
        default=0.0,
        help="Retain dual-scope evidence for an implicit ownership disclosure above this ratio.",
    )
    parser.add_argument(
        "--count-scope-fallback",
        action="store_true",
        help="Retain ranked opposite scope when a one-year entity-count preferred scope was catalog-backfilled.",
    )
    parser.add_argument(
        "--catalog-universe-scan",
        action="store_true",
        help="Expand an explicit broad unnamed-company scan to the bounded year/scope catalog universe.",
    )
    parser.add_argument("--universe-cap", type=int, default=200)
    parser.add_argument(
        "--common-evidence-year",
        action="store_true",
        help="Infer an omitted multi-entity year from strongest complete preferred-scope evidence.",
    )
    parser.add_argument("--evidence-year-pool", type=int, default=5000)
    args = parser.parse_args()

    rows = json.loads((args.candidate / "submission.json").read_text(encoding="utf-8"))
    reports, groups = _load_report_meta()
    retrieve_decomposed("warm")
    modes = {}
    for per_pair in args.per_pair:
        macro_precision = []
        macro_recall = []
        macro_f2 = []
        predicted_counts = []
        exact = 0
        misses = []
        reasons = Counter()
        for position, row in enumerate(rows, 1):
            question = str(row["question"])
            targets = set(_unique(row.get("relevant_docs", [])))
            facets = extract_all_facets(question)
            hits = retrieve_decomposed(
                question,
                per_pair=per_pair,
                pool=args.pool,
                cap=args.cap,
                backfill_unique_preferred_scope=args.facet_backfill,
                backfill_unique_scope_neutral=args.scope_neutral_backfill,
                backfill_explicit_opening_year=args.opening_year_backfill,
                backfill_financial_formula_year=args.formula_year_backfill,
                backfill_growth_scan_year=args.growth_scan_year_backfill,
                backfill_comparative_selection_year=(
                    args.comparative_selection_year_backfill
                ),
                backfill_comparative_series_year=(
                    args.comparative_series_year_backfill
                ),
                backfill_sparse_series_year=args.sparse_series_year_backfill,
                retain_ambiguous_series_reports=args.ambiguous_series_reports,
                implicit_scope_minimum_ratio=args.implicit_scope_minimum_ratio,
                implicit_ownership_scope_minimum_ratio=(
                    args.implicit_ownership_scope_minimum_ratio
                ),
                retain_count_scope_fallback=args.count_scope_fallback,
                expand_catalog_universe=args.catalog_universe_scan,
                universe_cap=args.universe_cap,
                infer_common_evidence_year=args.common_evidence_year,
                evidence_year_pool=args.evidence_year_pool,
            )
            predictions = _unique(
                str(hit["table_ref"]).split("|", 1)[0] for hit in hits
            )
            predicted = set(predictions)
            shared = len(targets.intersection(predicted))
            precision = shared / len(predicted) if predicted else 0.0
            recall = shared / len(targets) if targets else 0.0
            macro_precision.append(precision)
            macro_recall.append(recall)
            macro_f2.append(_f2(precision, recall))
            predicted_counts.append(len(predicted))
            exact += int(targets == predicted)
            if recall < 1.0:
                missing = sorted(targets - predicted)
                classified = [
                    {
                        "document": target,
                        "reason": _diagnose(
                            target, predictions, facets, reports, groups
                        ),
                        "group_size": len(
                            groups.get(
                                (
                                    reports.get(target, {}).get("ticker", ""),
                                    reports.get(target, {}).get("year", ""),
                                    reports.get(target, {}).get("scope", ""),
                                ),
                                [],
                            )
                        ),
                    }
                    for target in missing
                ]
                reasons.update(item["reason"] for item in classified)
                misses.append(
                    {
                        "id": int(row["id"]),
                        "question": question,
                        "facets": facets,
                        "targets": sorted(targets),
                        "predictions": predictions,
                        "missing": classified,
                    }
                )
            if position % 200 == 0:
                print(
                    "per_pair={0}: {1}/{2}".format(per_pair, position, len(rows)),
                    flush=True,
                )
        count = len(rows)
        modes[str(per_pair)] = {
            "questions": count,
            "macro_precision": round(sum(macro_precision) / count, 6),
            "macro_recall": round(sum(macro_recall) / count, 6),
            "macro_f2": round(sum(macro_f2) / count, 6),
            "exact_document_sets": exact,
            "missed_questions": len(misses),
            "average_predicted_documents": round(sum(predicted_counts) / count, 6),
            "reason_counts": dict(sorted(reasons.items())),
            "misses": misses,
        }

    payload = {
        "kind": "source_proven_document_retrieval_regression_not_btc_gold",
        "candidate": args.candidate.name,
        "cap": args.cap,
        "pool": args.pool,
        "facet_backfill": args.facet_backfill,
        "scope_neutral_backfill": args.scope_neutral_backfill,
        "opening_year_backfill": args.opening_year_backfill,
        "formula_year_backfill": args.formula_year_backfill,
        "growth_scan_year_backfill": args.growth_scan_year_backfill,
        "comparative_selection_year_backfill": (
            args.comparative_selection_year_backfill
        ),
        "comparative_series_year_backfill": (
            args.comparative_series_year_backfill
        ),
        "sparse_series_year_backfill": args.sparse_series_year_backfill,
        "ambiguous_series_reports": args.ambiguous_series_reports,
        "implicit_scope_minimum_ratio": args.implicit_scope_minimum_ratio,
        "implicit_ownership_scope_minimum_ratio": (
            args.implicit_ownership_scope_minimum_ratio
        ),
        "count_scope_fallback": args.count_scope_fallback,
        "catalog_universe_scan": args.catalog_universe_scan,
        "universe_cap": args.universe_cap,
        "common_evidence_year": args.common_evidence_year,
        "evidence_year_pool": args.evidence_year_pool,
        "modes": modes,
        "claim_limits": [
            "Targets are executable source bindings, not BTC hidden-gold labels.",
            "Metrics diagnose product retrieval only and are not leaderboard scores.",
            "A higher recall mode must also be evaluated for precision and downstream table context.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        key: {field: value for field, value in mode.items() if field != "misses"}
        for key, mode in modes.items()
    }
    print(json.dumps({"output": str(args.output), "modes": compact}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
