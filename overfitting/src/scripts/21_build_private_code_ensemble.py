"""Merge two or three independently verified code-mode runs into a base."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.codegen.consensus import (
    choose_code_ensemble_candidates,
    code_evidence,
    verify_code_consensus_candidate,
)
from vifinqa.codegen.generate import QuestionBundle
from vifinqa.extraction.build_store import Store
from vifinqa.utils.io import read_jsonl, setup_stdout, write_jsonl


def _by_id(path: str, label: str) -> dict[int, dict]:
    rows = read_jsonl(path)
    out = {int(row["id"]): row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"{label} contains duplicate ids")
    return out


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate-a", required=True)
    parser.add_argument("--candidate-b", required=True)
    parser.add_argument("--retrieval-a", required=True)
    parser.add_argument("--retrieval-b", required=True)
    parser.add_argument("--candidate-c", default="")
    parser.add_argument("--retrieval-c", default="")
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-out", default="")
    parser.add_argument("--expected-samples", type=int, default=5)
    parser.add_argument("--min-votes", type=int, default=4)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base)
    base = {int(row["id"]): row for row in base_rows}
    candidate_a = _by_id(args.candidate_a, "candidate A")
    candidate_b = _by_id(args.candidate_b, "candidate B")
    retrieval_a = _by_id(args.retrieval_a, "retrieval A")
    retrieval_b = _by_id(args.retrieval_b, "retrieval B")
    if bool(args.candidate_c) != bool(args.retrieval_c):
        raise ValueError("candidate C and retrieval C must be provided together")
    candidates = [candidate_a, candidate_b]
    retrievals = [retrieval_a, retrieval_b]
    labels = ["a", "b"]
    if args.candidate_c:
        candidates.append(_by_id(args.candidate_c, "candidate C"))
        retrievals.append(_by_id(args.retrieval_c, "retrieval C"))
        labels.append("c")
    expected_ids = set(base)
    if any(set(rows) != expected_ids for rows in (*candidates, *retrievals)):
        raise ValueError("ensemble inputs have different id sets")

    store = Store(Path(args.store_dir), cache_size=120)
    merged, audit, accepted = [], [], []
    reasons = Counter()
    for base_row in base_rows:
        qid = int(base_row["id"])
        bundles = [QuestionBundle(rows[qid], store, args.k)
                   for rows in retrievals]
        rows = [items[qid] for items in candidates]
        decisions = [
            verify_code_consensus_candidate(
                row, bundle, expected_samples=args.expected_samples,
                min_votes=args.min_votes)
            for row, bundle in zip(rows, bundles)
        ]
        choice = choose_code_ensemble_candidates(
            base_row, list(zip(rows, decisions, bundles)),
            expected_samples=args.expected_samples,
        )
        reasons[choice.reason] += 1
        changed = choice.candidate is not None
        if changed:
            replacement = dict(choice.candidate)
            replacement["source"] = f"private_code_ensemble:{replacement.get('source')}"
            replacement["detail"] = (
                f"{choice.reason}; replaced {base_row.get('source')}; "
                f"{replacement.get('detail', '')}"
            )
            merged.append(replacement)
            accepted.append(qid)
        else:
            merged.append(base_row)

        audit_row = {
            "id": qid,
            "accepted": changed,
            "reason": choice.reason,
            "base_answer": base_row.get("answer"),
        }
        for label, row, decision, bundle in zip(
                labels, rows, decisions, bundles):
            evidence = code_evidence(row, bundle)
            audit_row.update({
                f"decision_{label}": decision.reason,
                f"candidate_{label}_answer": row.get("answer"),
                f"candidate_{label}_votes": row.get("votes"),
                f"candidate_{label}_cells": sorted(evidence.cells),
            })
        audit.append(audit_row)

    write_jsonl(args.out, merged)
    audit_path = args.audit_out or f"{args.out}.audit.jsonl"
    write_jsonl(audit_path, audit)
    print(f"code ensemble {len(merged)} rows; changed={len(accepted)} ids={accepted}")
    print("reasons:", dict(reasons))
    print(f"audit -> {audit_path}")


if __name__ == "__main__":
    main()
