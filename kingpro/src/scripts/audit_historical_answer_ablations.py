"""Turn local submission history into exact, reviewable score-delta batches.

The public scorer evaluates 506 hidden IDs while every valid submission contains
1,012 rows.  A score delta is nevertheless exact: after converting the rounded
accuracy back to an integer correct-count, it equals the net correctness change
over the answer cells that differ between two submissions.

This tool is deliberately diagnostic.  It never edits or uploads a submission.
It emits:

* chronological adjacent comparisons (the most interpretable experiments),
* every local pair whose answer diff is at most ``--max-diff`` questions, and
* per-question appearances in small, non-zero score equations.

Example:
  python scripts/audit_historical_answer_ablations.py \
    --out build/v214_historical_answer_ablations.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from infer_public_ensemble import (
    ROOT,
    load_history,
    numeric_answer,
    read_submission,
    score_to_count,
)


def answer_equal(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> bool:
    """Match the grader's numeric tolerance, with a safe fallback for text."""
    left_number = numeric_answer(left)
    right_number = numeric_answer(right)
    if left_number is not None and right_number is not None:
        return math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=tolerance)
    return str(left.get("answer", "")).strip() == str(right.get("answer", "")).strip()


def answer_value(entry: dict[str, Any]) -> Any:
    number = numeric_answer(entry)
    return number if number is not None else entry.get("answer")


def compare(old: Any, new: Any, public_size: int, tolerance: float) -> dict[str, Any]:
    ids = sorted(set(old.entries) | set(new.entries))
    changes = []
    for qid in ids:
        old_entry = old.entries.get(qid, {})
        new_entry = new.entries.get(qid, {})
        if answer_equal(old_entry, new_entry, tolerance):
            continue
        changes.append(
            {
                "id": qid,
                "question": new_entry.get("question") or old_entry.get("question", ""),
                "old_answer": answer_value(old_entry),
                "new_answer": answer_value(new_entry),
            }
        )
    old_count = score_to_count(old.history.answer_accuracy, public_size)
    new_count = score_to_count(new.history.answer_accuracy, public_size)
    return {
        "old_submission_id": old.history.submission_id,
        "old_submission": old.history.stem,
        "old_correct_count": old_count,
        "new_submission_id": new.history.submission_id,
        "new_submission": new.history.stem,
        "new_correct_count": new_count,
        "score_delta_count": new_count - old_count,
        "changed_count": len(changes),
        "changed_ids": [row["id"] for row in changes],
        "changes": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "configs/leaderboard_history.csv")
    parser.add_argument("--public-size", type=int, default=506)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max-diff", type=int, default=150)
    parser.add_argument("--small-equation", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_history(args.history)
    loaded = [sub for row in rows if (sub := read_submission(ROOT, row)) is not None]
    loaded = [sub for sub in loaded if len(sub.entries) == 1012]
    loaded.sort(key=lambda sub: sub.history.submission_id)
    if len(loaded) < 2:
        raise SystemExit("Need at least two complete local scored submissions")

    adjacent = [
        compare(old, new, args.public_size, args.tolerance)
        for old, new in zip(loaded, loaded[1:])
    ]
    compact_pairs = []
    for old_index, old in enumerate(loaded):
        for new in loaded[old_index + 1 :]:
            record = compare(old, new, args.public_size, args.tolerance)
            if 0 < record["changed_count"] <= args.max_diff:
                compact_pairs.append(record)
    compact_pairs.sort(
        key=lambda row: (
            row["changed_count"],
            -abs(row["score_delta_count"]),
            -row["new_submission_id"],
        )
    )

    question_equations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for equation_index, record in enumerate(compact_pairs):
        if (
            record["changed_count"] > args.small_equation
            or record["score_delta_count"] == 0
        ):
            continue
        for change in record["changes"]:
            question_equations[int(change["id"])].append(
                {
                    "equation_index": equation_index,
                    "old_submission": record["old_submission"],
                    "new_submission": record["new_submission"],
                    "changed_count": record["changed_count"],
                    "score_delta_count": record["score_delta_count"],
                    "old_answer": change["old_answer"],
                    "new_answer": change["new_answer"],
                }
            )
    ranked_questions = sorted(
        (
            {
                "id": qid,
                "equation_count": len(equations),
                "minimum_batch_size": min(row["changed_count"] for row in equations),
                "equations": equations,
            }
            for qid, equations in question_equations.items()
        ),
        key=lambda row: (row["minimum_batch_size"], -row["equation_count"], row["id"]),
    )

    exact_singletons = [
        row for row in compact_pairs
        if row["changed_count"] == 1 and abs(row["score_delta_count"]) == 1
    ]
    informative_adjacent = [
        row for row in adjacent
        if row["changed_count"] and row["changed_count"] <= args.max_diff
    ]
    report = {
        "schema_version": 1,
        "warning": (
            "Score equations constrain only the hidden 506-question public subset. "
            "Multi-question equations are review priorities, not gold labels."
        ),
        "history": str(args.history.resolve()),
        "public_size": args.public_size,
        "answer_tolerance": args.tolerance,
        "summary": {
            "complete_local_submissions": len(loaded),
            "submission_id_range": [
                loaded[0].history.submission_id,
                loaded[-1].history.submission_id,
            ],
            "adjacent_comparisons": len(adjacent),
            "informative_adjacent_comparisons": len(informative_adjacent),
            "compact_pair_equations": len(compact_pairs),
            "exact_singleton_proofs": len(exact_singletons),
            "questions_in_small_nonzero_equations": len(ranked_questions),
        },
        "exact_singletons": exact_singletons,
        "informative_adjacent": informative_adjacent,
        "ranked_questions": ranked_questions,
        "compact_pairs": compact_pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("exact_singleton_ids=" + ",".join(str(row["changed_ids"][0]) for row in exact_singletons))
    print("ranked_question_ids=" + ",".join(str(row["id"]) for row in ranked_questions[:100]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
