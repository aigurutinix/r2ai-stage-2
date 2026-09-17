"""Audit BM25 evidence for internal predecessor years in sparse series.

Targets are executable source bindings, not BTC hidden gold.  The report is a
threshold diagnostic only; question IDs are emitted for review and are never
permitted as retrieval features.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.retrieval.bm25_index import (  # noqa: E402
    _load_index,
    extract_all_facets,
    sparse_series_predecessor_years,
    tokenize,
)


DEFAULT_CANDIDATE = ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
DEFAULT_OUTPUT = ROOT / "build" / "demo_compliance" / "sparse_series_year_competition_v1.json"


def _report_id(table_ref: str) -> str:
    return str(table_ref).split("|", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool", type=int, default=1000)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    )
    args = parser.parse_args()

    rows = json.loads((args.candidate / "submission.json").read_text(encoding="utf-8"))
    retriever, meta = _load_index()
    report_meta = {}
    for item in meta:
        report_meta.setdefault(
            _report_id(item["table_ref"]),
            {
                "ticker": str(item["ticker"]),
                "year": str(item["year"]),
                "scope": str(item["scope"]),
            },
        )

    observations = []
    eligible_questions = 0
    for position, row in enumerate(rows, 1):
        question = str(row["question"])
        facets = extract_all_facets(question)
        candidates = sparse_series_predecessor_years(
            facets["years"],
            entity_count=len(facets["tickers"]),
            analytic=bool(facets["analytic"]),
        )
        if not candidates:
            continue
        eligible_questions += 1
        ticker = str(facets["tickers"][0])
        scope = str(facets["scope"])
        idx, scores = retriever.retrieve(
            [tokenize(question)], k=min(args.pool, len(meta)), show_progress=False
        )
        best_by_year_scope: dict[tuple[str, str], dict] = {}
        for rank, (index, score) in enumerate(
            zip(idx[0].tolist(), scores[0].tolist()), 1
        ):
            item = meta[index]
            item_scope = str(item["scope"])
            if str(item["ticker"]) != ticker or item_scope not in {scope, "không rõ"}:
                continue
            year = str(item["year"])
            key = (year, item_scope)
            current = best_by_year_scope.get(key)
            if current is None or float(score) > current["score"]:
                best_by_year_scope[key] = {
                    "report_id": _report_id(item["table_ref"]),
                    "score": float(score),
                    "rank": rank,
                    "scope": item_scope,
                }
        best_by_year = {}
        for year in {*facets["years"], *candidates}:
            preferred = best_by_year_scope.get((year, scope))
            neutral = best_by_year_scope.get((year, "không rõ"))
            if preferred is not None:
                best_by_year[year] = preferred
            elif neutral is not None:
                best_by_year[year] = neutral
        explicit_scores = [
            best_by_year[year]["score"]
            for year in facets["years"]
            if year in best_by_year
        ]
        if not explicit_scores:
            continue
        target_docs = {str(value) for value in row.get("relevant_docs", [])}
        target_years = {
            report_meta[report_id]["year"]
            for report_id in target_docs
            if report_id in report_meta
        }
        maximum = max(explicit_scores)
        mean = statistics.fmean(explicit_scores)
        for candidate_year in candidates:
            evidence = best_by_year.get(candidate_year)
            if evidence is None:
                continue
            observations.append(
                {
                    "id": int(row["id"]),
                    "question": question,
                    "ticker": ticker,
                    "scope": scope,
                    "explicit_years": facets["years"],
                    "candidate_year": candidate_year,
                    "candidate_report": evidence["report_id"],
                    "candidate_scope": evidence["scope"],
                    "candidate_score": evidence["score"],
                    "candidate_rank": evidence["rank"],
                    "candidate_year_is_target": candidate_year in target_years,
                    "ratio_to_max_explicit": evidence["score"] / maximum if maximum else 0.0,
                    "ratio_to_mean_explicit": evidence["score"] / mean if mean else 0.0,
                }
            )
        if position % 200 == 0:
            print(f"scanned {position}/{len(rows)}", flush=True)

    summaries = {}
    for threshold in sorted(set(args.thresholds)):
        selected = [
            item for item in observations if item["ratio_to_max_explicit"] >= threshold
        ]
        summaries[str(threshold)] = {
            "candidate_years_selected": len(selected),
            "target_years_selected": sum(item["candidate_year_is_target"] for item in selected),
            "non_target_years_selected": sum(
                not item["candidate_year_is_target"] for item in selected
            ),
            "questions_affected": len({item["id"] for item in selected}),
        }

    payload = {
        "kind": "source_bound_sparse_series_year_score_diagnostic_not_btc_gold",
        "candidate": args.candidate.name,
        "pool": args.pool,
        "eligible_questions": eligible_questions,
        "observations": len(observations),
        "thresholds": summaries,
        "target_observations": [
            item for item in observations if item["candidate_year_is_target"]
        ],
        "competitive_observations": [
            item
            for item in observations
            if item["ratio_to_max_explicit"] >= min(args.thresholds)
        ],
        "claim_limits": [
            "Targets are executable source bindings, not BTC hidden-gold labels.",
            "Question IDs are audit labels only and are not retrieval features.",
            "Any threshold requires a full 1,012-question precision/recall/F2 regression.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **summaries}, indent=2))


if __name__ == "__main__":
    main()
