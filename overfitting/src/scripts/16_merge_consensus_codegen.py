"""Merge only strictly verified N-sample LLM challengers into a checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.codegen.consensus import verify_consensus_candidate
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


def _read_ids(path: str) -> set[int] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text.replace(",", " ").split()
    if isinstance(value, dict):
        value = value.get("ids", [])
    if not isinstance(value, list):
        raise ValueError("ids file must contain a JSON list/object or text ids")
    return {int(item) for item in value}


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit-out", default="")
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--base-source-token", default="")
    parser.add_argument("--expected-samples", type=int, default=5)
    parser.add_argument("--min-votes", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=78.0)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base)
    base = _by_id(base_rows, "base")
    candidate = _by_id(read_jsonl(args.candidate), "candidate")
    retrieval = _by_id(read_jsonl(args.retrieval), "retrieval")
    if set(base) != set(candidate) or set(base) != set(retrieval):
        raise ValueError("base, candidate and retrieval id sets differ")

    allow_ids = _read_ids(args.ids_file)
    store = Store(Path(args.store_dir), cache_size=120)
    merged, audit, accepted = [], [], []
    reasons = Counter()
    for row in base_rows:
        qid = int(row["id"])
        eligible = allow_ids is None or qid in allow_ids
        if args.base_source_token:
            eligible = eligible and args.base_source_token in str(row.get("source") or "")
        if not eligible:
            merged.append(row)
            continue

        challenger = candidate[qid]
        bundle = QuestionBundle(retrieval[qid], store, args.k)
        decision = verify_consensus_candidate(
            challenger, bundle,
            expected_samples=args.expected_samples,
            min_votes=args.min_votes,
            min_confidence=args.min_confidence,
        )
        reasons[decision.reason] += 1
        changed = False
        if decision.accepted:
            replacement = dict(challenger)
            replacement["source"] = f"private_consensus:{challenger.get('source')}"
            replacement["detail"] = (
                f"{decision.reason}; replaced {row.get('source')}; "
                f"{challenger.get('detail', '')}"
            )
            merged.append(replacement)
            accepted.append(qid)
            changed = round(float(replacement["answer"]), 2) != round(
                float(row.get("answer") or 0.0), 2)
        else:
            merged.append(row)
        audit.append({
            "id": qid,
            "accepted": decision.accepted,
            "changed_answer": changed,
            "reason": decision.reason,
            "base_answer": row.get("answer"),
            "candidate_answer": challenger.get("answer"),
            "votes": challenger.get("votes"),
            "n_ok": challenger.get("n_ok"),
        })

    write_jsonl(args.out, merged)
    audit_path = args.audit_out or f"{args.out}.audit.jsonl"
    write_jsonl(audit_path, audit)
    print(f"merged {len(merged)} rows; accepted={len(accepted)} ids={accepted}")
    print(f"changed answers={sum(item['changed_answer'] for item in audit)}")
    print("reasons:", dict(reasons))
    print(f"audit -> {audit_path}")


if __name__ == "__main__":
    main()
