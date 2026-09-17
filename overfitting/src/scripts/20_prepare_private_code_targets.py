"""Build the unresolved private-test cohort for independent code-mode runs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.utils.io import read_jsonl, setup_stdout


def _by_id(path: str) -> dict[int, dict]:
    rows = read_jsonl(path)
    out = {int(row["id"]): row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"duplicate ids in {path}")
    return out


def build_target_cohort(
    checkpoint: dict[int, dict],
    retrieval: dict[int, dict],
    audits: list[dict[int, dict]],
) -> tuple[list[int], dict]:
    """Return failed/single-vote ids not accepted by any strict verifier."""
    expected = set(checkpoint)
    if set(retrieval) != expected:
        raise ValueError("checkpoint and retrieval id sets differ")
    if any(not set(audit).issubset(expected) for audit in audits):
        raise ValueError("an audit contains ids outside the checkpoint")

    weak = {
        qid for qid, row in checkpoint.items()
        if row.get("status") != "ok"
        or (
            row.get("source") == "llm_select"
            and int(row.get("votes") or 0) == 1
            and int(row.get("n_ok") or 0) == 1
        )
    }
    accepted = {
        qid for audit in audits for qid, row in audit.items()
        if row.get("accepted")
    }
    target_ids = sorted(weak - accepted)

    ops = Counter()
    output_types = Counter()
    no_requirements = 0
    failed = 0
    for qid in target_ids:
        row = checkpoint[qid]
        route = retrieval[qid].get("route") or {}
        ops[str((route.get("plan") or {}).get("op") or "lookup")] += 1
        output_types[str(route.get("output_type") or "number")] += 1
        no_requirements += not bool(route.get("evidence_requirements"))
        failed += row.get("status") != "ok"

    stats = {
        "weak_or_failed": len(weak),
        "accepted_by_prior_verifiers": len(weak & accepted),
        "target_count": len(target_ids),
        "failed": failed,
        "no_canonical_requirements": no_requirements,
        "operations": dict(ops),
        "output_types": dict(output_types),
    }
    return target_ids, stats


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--audit", action="append", default=[], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = _by_id(args.checkpoint)
    retrieval = _by_id(args.retrieval)
    audits = [_by_id(path) for path in args.audit]
    target_ids, stats = build_target_cohort(checkpoint, retrieval, audits)
    payload = {
        "name": "private_v33_unresolved_code_challengers",
        "count": len(target_ids),
        "stats": stats,
        "ids": target_ids,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"target ids -> {args.out}")


if __name__ == "__main__":
    main()
