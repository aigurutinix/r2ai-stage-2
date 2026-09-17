"""Measure preferred/opposite report-scope BM25 competition on source bindings.

This is a diagnostic for choosing a content-score threshold.  It uses no BTC
hidden labels and does not modify retrieval or submission artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.retrieval.bm25_index import (  # noqa: E402
    _load_index,
    extract_all_facets,
    has_explicit_scope_cue,
    tokenize,
)


DEFAULT_CANDIDATE = ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
DEFAULT_OUTPUT = ROOT / "build" / "demo_compliance" / "implicit_scope_competition_v1.json"


def _report_id(table_ref: str) -> str:
    return str(table_ref).split("|", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool", type=int, default=1000)
    parser.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=[],
        help="Optional audit-only question IDs; never used by retrieval policy.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    )
    args = parser.parse_args()

    rows = json.loads((args.candidate / "submission.json").read_text(encoding="utf-8"))
    retriever, meta = _load_index()
    facet_reports: dict[tuple[str, str, str], set[str]] = {}
    for item in meta:
        key = (str(item["ticker"]), str(item["year"]), str(item["scope"]))
        facet_reports.setdefault(key, set()).add(_report_id(item["table_ref"]))
    observations = []
    backfilled_preferred_observations = []
    for position, row in enumerate(rows, 1):
        if args.ids and int(row["id"]) not in set(args.ids):
            continue
        question = str(row["question"])
        facets = extract_all_facets(question)
        if (
            has_explicit_scope_cue(question)
            or not facets["tickers"]
            or not facets["years"]
        ):
            continue
        idx, scores = retriever.retrieve(
            [tokenize(question)], k=min(args.pool, len(meta)), show_progress=False
        )
        target_docs = {str(value) for value in row.get("relevant_docs", [])}
        for ticker in facets["tickers"]:
            for year in facets["years"]:
                preferred_scope = facets["scope"]
                opposite_scope = (
                    "công ty mẹ" if preferred_scope == "hợp nhất" else "hợp nhất"
                )
                best: dict[str, dict] = {}
                for rank, (index, score) in enumerate(
                    zip(idx[0].tolist(), scores[0].tolist()), 1
                ):
                    candidate = meta[index]
                    if candidate["ticker"] != ticker or candidate["year"] != year:
                        continue
                    scope = str(candidate["scope"])
                    if scope not in {preferred_scope, opposite_scope}:
                        continue
                    report_id = _report_id(candidate["table_ref"])
                    current = best.get(scope)
                    if current is None or float(score) > current["score"]:
                        best[scope] = {
                            "report_id": report_id,
                            "score": float(score),
                            "rank": rank,
                        }
                preferred = best.get(preferred_scope)
                opposite = best.get(opposite_scope)
                if not opposite:
                    continue
                if not preferred:
                    preferred_catalog = sorted(
                        facet_reports.get((ticker, year, preferred_scope), set())
                    )
                    if len(preferred_catalog) == 1:
                        backfilled_preferred_observations.append(
                            {
                                "id": int(row["id"]),
                                "question": question,
                                "analytic": bool(facets["analytic"]),
                                "entity_count": len(facets["tickers"]),
                                "year_count": len(facets["years"]),
                                "ticker": ticker,
                                "year": year,
                                "preferred_scope": preferred_scope,
                                "preferred_report": preferred_catalog[0],
                                "preferred_is_target": preferred_catalog[0] in target_docs,
                                "opposite_scope": opposite_scope,
                                "opposite_report": opposite["report_id"],
                                "opposite_score": opposite["score"],
                                "opposite_rank": opposite["rank"],
                                "opposite_is_target": opposite["report_id"] in target_docs,
                            }
                        )
                    continue
                if preferred["score"] <= 0:
                    continue
                observations.append(
                    {
                        "id": int(row["id"]),
                        "question": question,
                        "analytic": bool(facets["analytic"]),
                        "entity_count": len(facets["tickers"]),
                        "year_count": len(facets["years"]),
                        "ticker": ticker,
                        "year": year,
                        "preferred_scope": preferred_scope,
                        "preferred_report": preferred["report_id"],
                        "preferred_score": preferred["score"],
                        "preferred_rank": preferred["rank"],
                        "preferred_is_target": preferred["report_id"] in target_docs,
                        "opposite_scope": opposite_scope,
                        "opposite_report": opposite["report_id"],
                        "opposite_score": opposite["score"],
                        "opposite_rank": opposite["rank"],
                        "opposite_is_target": opposite["report_id"] in target_docs,
                        "opposite_to_preferred_ratio": opposite["score"] / preferred["score"],
                    }
                )
        if position % 200 == 0:
            print(f"scanned {position}/{len(rows)}", flush=True)

    threshold_summary = {}
    for threshold in sorted(set(args.thresholds)):
        selected = [
            item
            for item in observations
            if item["opposite_to_preferred_ratio"] >= threshold
        ]
        threshold_summary[str(threshold)] = {
            "opposite_reports_added": len(selected),
            "target_reports_recovered": sum(item["opposite_is_target"] for item in selected),
            "non_target_reports_added": sum(not item["opposite_is_target"] for item in selected),
            "preferred_target_pairs": sum(item["preferred_is_target"] for item in selected),
        }

    payload = {
        "kind": "source_bound_implicit_scope_score_diagnostic_not_btc_gold",
        "candidate": args.candidate.name,
        "pool": args.pool,
        "observations": len(observations),
        "backfilled_preferred_observations": backfilled_preferred_observations,
        "backfilled_preferred_summary": {
            "pairs": len(backfilled_preferred_observations),
            "target_opposite_reports": sum(
                item["opposite_is_target"] for item in backfilled_preferred_observations
            ),
            "non_target_opposite_reports": sum(
                not item["opposite_is_target"] for item in backfilled_preferred_observations
            ),
            "preferred_target_pairs": sum(
                item["preferred_is_target"] for item in backfilled_preferred_observations
            ),
        },
        "thresholds": threshold_summary,
        "target_opposite_scope_observations": [
            item for item in observations if item["opposite_is_target"]
        ],
        "competitive_observations": [
            item
            for item in observations
            if item["opposite_to_preferred_ratio"] >= min(args.thresholds)
        ],
        "claim_limits": [
            "Targets are executable source bindings, not BTC hidden-gold labels.",
            "The diagnostic selects a general score threshold, never a question ID.",
            "Any threshold still requires a full 1,012-question retrieval regression.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **threshold_summary}, indent=2))


if __name__ == "__main__":
    main()
