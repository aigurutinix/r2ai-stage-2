"""Build a conservative final ensemble from two targeted Qwen challenger runs."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.codegen.consensus import (
    choose_ensemble_candidate,
    verify_consensus_candidate,
)
from vifinqa.codegen.generate import QuestionBundle
from vifinqa.extraction.build_store import Store
from vifinqa.utils.io import read_jsonl, setup_stdout, write_jsonl


def _by_id(rows: list[dict], label: str) -> dict[int, dict]:
    out = {}
    for row in rows:
        qid = int(row["id"])
        if qid in out:
            raise ValueError(f"{label} contains duplicate id {qid}")
        out[qid] = row
    return out


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="codegen checkpoint with the best observed private score")
    parser.add_argument("--candidate-a", required=True)
    parser.add_argument("--candidate-b", required=True)
    parser.add_argument("--retrieval-a", required=True)
    parser.add_argument("--retrieval-b", required=True)
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-out", default="")
    parser.add_argument("--expected-samples", type=int, default=5)
    parser.add_argument("--min-votes", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=78.0)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base)
    base = _by_id(base_rows, "base")
    candidate_a = _by_id(read_jsonl(args.candidate_a), "candidate A")
    candidate_b = _by_id(read_jsonl(args.candidate_b), "candidate B")
    retrieval_a = _by_id(read_jsonl(args.retrieval_a), "retrieval A")
    retrieval_b = _by_id(read_jsonl(args.retrieval_b), "retrieval B")
    expected_ids = set(base)
    if any(set(value) != expected_ids for value in (
            candidate_a, candidate_b, retrieval_a, retrieval_b)):
        raise ValueError("ensemble inputs have different id sets")

    store = Store(Path(args.store_dir), cache_size=120)
    merged, audit, accepted = [], [], []
    reasons = Counter()
    for row in base_rows:
        qid = int(row["id"])
        candidates = []
        for challenger, retrieval in (
                (candidate_a[qid], retrieval_a[qid]),
                (candidate_b[qid], retrieval_b[qid])):
            bundle = QuestionBundle(retrieval, store, args.k)
            decision = verify_consensus_candidate(
                challenger, bundle,
                expected_samples=args.expected_samples,
                min_votes=args.min_votes,
                min_confidence=args.min_confidence,
            )
            candidates.append((challenger, decision))
        choice = choose_ensemble_candidate(
            row, candidates, expected_samples=args.expected_samples)
        reasons[choice.reason] += 1
        if choice.candidate is None:
            merged.append(row)
        else:
            replacement = dict(choice.candidate)
            replacement["source"] = f"private_ensemble:{replacement.get('source')}"
            replacement["detail"] = (
                f"{choice.reason}; replaced {row.get('source')}; "
                f"{replacement.get('detail', '')}"
            )
            merged.append(replacement)
            accepted.append(qid)
        audit.append({
            "id": qid, "accepted": choice.candidate is not None,
            "reason": choice.reason, "base_answer": row.get("answer"),
            "candidate_a_answer": candidate_a[qid].get("answer"),
            "candidate_b_answer": candidate_b[qid].get("answer"),
            "candidate_a_votes": candidate_a[qid].get("votes"),
            "candidate_b_votes": candidate_b[qid].get("votes"),
        })

    write_jsonl(args.out, merged)
    audit_path = args.audit_out or f"{args.out}.audit.jsonl"
    write_jsonl(audit_path, audit)
    print(f"ensemble {len(merged)} rows; accepted={len(accepted)} ids={accepted}")
    print("reasons:", dict(reasons))
    print(f"audit -> {audit_path}")


if __name__ == "__main__":
    main()
