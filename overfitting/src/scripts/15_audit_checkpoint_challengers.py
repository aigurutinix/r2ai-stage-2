"""Build a shadow challenger matrix for successful single-vote LLM rows.

No checkpoint row is overwritten. The output contains a full-size candidate
JSONL suitable for an explicit allowlisted ``scripts/11_merge_codegen.py`` run,
plus a compact audit matrix and tiered allowlists for controlled ablations.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.codegen.exact_average import try_exact_average_answer
from vifinqa.codegen.exact_difference import try_exact_difference_answer
from vifinqa.codegen.exact_growth import try_exact_growth_answer
from vifinqa.codegen.exact_lookup import try_exact_lookup_answer
from vifinqa.codegen.exact_ranking import try_exact_ranking_answer
from vifinqa.codegen.exact_ratio import try_exact_ratio_answer
from vifinqa.codegen.generate import QuestionBundle, _final, _run_validated
from vifinqa.extraction.build_store import Store
from vifinqa.utils.io import ensure_dir, read_jsonl, setup_stdout, write_jsonl


def _by_id(rows: list[dict], label: str) -> dict[int, dict]:
    indexed = {}
    for row in rows:
        qid = int(row["id"])
        if qid in indexed:
            raise ValueError(f"{label} contains duplicate id {qid}")
        indexed[qid] = row
    return indexed


def _empty_candidate(row: dict, detail: str) -> dict:
    return {
        "id": int(row["id"]), "question": row["question"],
        "answer": 0.0, "pandas_query": "0.0", "used_vars": [],
        "status": "failed", "source": "challenger_none",
        "votes": 0, "n_ok": 0, "detail": detail,
        "detail_conf": 0.0, "semantic": {}, "run_signature": "",
    }


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, default=config.STORE_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=112)
    parser.add_argument("--source-token", default="llm")
    parser.add_argument(
        "--operation", choices=(
            "lookup", "difference", "ranking", "average", "growth_pct", "ratio"),
        default="lookup")
    parser.add_argument("--agreement-tolerance", type=float, default=0.011)
    args = parser.parse_args()

    retrieval = _by_id(read_jsonl(args.retrieval), "retrieval")
    checkpoint_rows = read_jsonl(args.checkpoint)
    checkpoint = _by_id(checkpoint_rows, "checkpoint")
    if set(retrieval) != set(checkpoint):
        raise ValueError("retrieval/checkpoint id sets differ")

    out_dir = ensure_dir(args.out_dir)
    store = Store(args.store_dir, cache_size=120)
    matrix = []
    candidates = []
    allowlists: dict[str, list[int]] = (
        {
            "agree": [], "vas_ranking_current_disagree": [],
            "vas_ranking_mixed_disagree": [], "note_ranking_exact_disagree": [],
        }
        if args.operation == "ranking"
        else
        {
            "agree": [], "vas_ratio_current_disagree": [],
            "vas_ratio_mixed_disagree": [], "note_ratio_exact_disagree": [],
        }
        if args.operation == "ratio"
        else
        {
            "agree": [], "vas_pair_current_disagree": [],
            "vas_pair_mixed_disagree": [], "note_pair_exact_disagree": [],
        }
        if args.operation == "difference"
        else {
            "agree": [], "vas_average_current_disagree": [],
            "vas_average_mixed_disagree": [],
            "note_average_exact_disagree": [],
        }
        if args.operation == "average"
        else
        {
            "agree": [], "vas_growth_current_disagree": [],
            "vas_growth_mixed_disagree": [],
            "note_growth_exact_disagree": [],
        }
        if args.operation == "growth_pct"
        else {
            "agree": [], "vas_current_disagree": [],
            "vas_prior_disagree": [], "note_exact_disagree": [],
        }
    )
    counts = Counter()

    for base in checkpoint_rows:
        qid = int(base["id"])
        rec = retrieval[qid]
        is_single_vote_llm = (
            args.source_token in str(base.get("source") or "")
            and int(base.get("votes", 0) or 0) == 1
            and int(base.get("n_ok", 0) or 0) == 1
        )
        if not is_single_vote_llm:
            candidates.append(_empty_candidate(base, "outside LLM single-vote cohort"))
            continue

        counts["llm_single_vote"] += 1
        bundle = QuestionBundle(rec, store, args.k)
        route_op = str((bundle.route.get("plan") or {}).get("op", "lookup"))
        if args.operation == "difference":
            exact = try_exact_difference_answer(bundle.route, bundle.tables)
        elif args.operation == "ranking":
            exact = try_exact_ranking_answer(bundle.route, bundle.tables)
        elif args.operation == "average":
            exact = try_exact_average_answer(bundle.route, bundle.tables)
        elif args.operation == "growth_pct":
            exact = try_exact_growth_answer(bundle.route, bundle.tables)
        elif args.operation == "ratio":
            exact = try_exact_ratio_answer(bundle.route, bundle.tables)
        else:
            exact = try_exact_lookup_answer(bundle.route, bundle.tables)
        candidate = _empty_candidate(base, exact.detail)
        agreement = "unresolved"

        if route_op == "lookup":
            counts["lookup"] += 1
        if exact.ok:
            executed = _run_validated(bundle, exact.pandas_query)
            if executed.get("status") == "ok":
                answer = round(float(executed["value"]), 2)
                candidate = _final(
                    bundle, answer, exact.pandas_query,
                    f"challenger_exact_{args.operation}",
                    semantic=executed.get("semantic"),
                )
                candidate["detail"] = exact.detail
                candidate["detail_conf"] = exact.confidence
                candidate["challenger_tier"] = exact.tier
                delta = abs(answer - float(base.get("answer", 0.0) or 0.0))
                agreement = (
                    "agree" if delta <= args.agreement_tolerance else "disagree")
                allow_key = (
                    "agree" if agreement == "agree"
                    else f"{exact.tier}_disagree"
                )
                allowlists[allow_key].append(qid)
                counts[f"exact_{agreement}"] += 1
                counts[f"tier_{exact.tier}_{agreement}"] += 1
            else:
                candidate["detail"] = (
                    f"execution rejected: {executed.get('detail') or executed}")
                counts["execution_rejected"] += 1
        else:
            counts[f"refused_{route_op}"] += 1

        candidates.append(candidate)
        matrix.append({
            "id": qid,
            "question": base["question"],
            "route_op": route_op,
            "output_type": bundle.route.get("output_type"),
            "baseline": {
                "answer": base.get("answer"), "source": base.get("source"),
                "votes": base.get("votes"), "n_ok": base.get("n_ok"),
            },
            "challenger": {
                "status": candidate.get("status"),
                "answer": candidate.get("answer"),
                "tier": candidate.get("challenger_tier", ""),
                "confidence": candidate.get("detail_conf", 0.0),
                "detail": candidate.get("detail", ""),
            },
            "agreement": agreement,
        })

    write_jsonl(out_dir / "challenger_matrix.jsonl", matrix)
    write_jsonl(out_dir / "challenger_codegen.jsonl", candidates)
    summary = {
        "retrieval": str(args.retrieval),
        "checkpoint": str(args.checkpoint),
        "rows": len(checkpoint_rows),
        "operation": args.operation,
        "counts": dict(sorted(counts.items())),
        "allowlists": allowlists,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    for name, ids in allowlists.items():
        print(f"{name}: {len(ids)} ids={ids}")
    print(f"audit -> {out_dir}")


if __name__ == "__main__":
    main()
