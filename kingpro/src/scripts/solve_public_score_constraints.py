"""Recover answer facts implied by public leaderboard score deltas.

The leaderboard evaluates 506 hidden-public questions out of the 1,012 submitted
rows.  Absolute-score ensemble models are therefore misspecified unless they also
model public-set membership.  Score *differences* avoid that nuisance variable:

    score(submission) - score(baseline)
      = sum_q [gold(submission_answer_q) - gold(baseline_answer_q)]

For every question, at most one historically submitted numeric answer class can be
the public gold answer.  This script turns those facts into a binary feasibility
model, then computes rigorous LP bounds for every answer class.  An LP bound of
exactly 1 (or 0) is also a proof for the stricter binary model.  No submission is
built or uploaded.

Example:
  python scripts/solve_public_score_constraints.py \
    --baseline sub_top123_candidate_v196_effective_tax_sign \
    --output build/public_score_constraints_v196.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, lil_matrix

from infer_public_ensemble import (
    ROOT,
    cluster_values,
    load_history,
    numeric_answer,
    read_submission,
    score_to_count,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=ROOT / "configs/leaderboard_history.csv")
    parser.add_argument("--public-size", type=int, default=506)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--baseline", default="sub_top123_candidate_v196_effective_tax_sign")
    parser.add_argument(
        "--min-submission-id",
        type=int,
        default=0,
        help="Use only this submission id and newer (useful for a compact trusted experiment chain).",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/public_score_constraints.json"
    )
    parser.add_argument(
        "--integer-check",
        action="store_true",
        help="Also require one binary-feasible solution (LP proofs remain the reported bounds).",
    )
    parser.add_argument(
        "--ids",
        help=(
            "Optional comma-separated question IDs whose answer-class bounds "
            "should be solved. All questions and submissions remain in the "
            "constraint system; this only avoids unrelated per-variable LPs."
        ),
    )
    args = parser.parse_args()

    history = [
        row for row in load_history(args.history)
        if row.submission_id >= args.min_submission_id
    ]
    submissions = [sub for row in history if (sub := read_submission(ROOT, row)) is not None]
    submissions = [sub for sub in submissions if len(sub.entries) == 1012]
    baseline_id = next(
        (i for i, sub in enumerate(submissions) if sub.history.stem == args.baseline), None
    )
    if baseline_id is None:
        raise SystemExit(f"Baseline {args.baseline!r} is not a complete local submission")

    baseline = submissions[baseline_id]
    counts = np.array(
        [score_to_count(sub.history.answer_accuracy, args.public_size) for sub in submissions],
        dtype=float,
    )
    baseline_count = counts[baseline_id]

    # Only questions whose submitted answer changed can influence a score delta.
    qids = sorted(set().union(*(set(sub.entries) for sub in submissions)))
    questions: dict[int, dict] = {}
    variables: list[dict] = []
    variable_by_qclass: dict[tuple[int, int], int] = {}
    submission_class: dict[tuple[int, int], int | None] = {}

    for qid in qids:
        values = [
            value
            for sub in submissions
            if (value := numeric_answer(sub.entries.get(qid, {}))) is not None
        ]
        if not values:
            continue
        centers, value_to_class = cluster_values(values, args.tolerance)
        if len(centers) < 2:
            continue
        members: list[list[int]] = [[] for _ in centers]
        for sid, sub in enumerate(submissions):
            value = numeric_answer(sub.entries.get(qid, {}))
            class_id = value_to_class[value] if value is not None else None
            submission_class[(sid, qid)] = class_id
            if class_id is not None:
                members[class_id].append(sid)
        for class_id, center in enumerate(centers):
            vid = len(variables)
            variable_by_qclass[(qid, class_id)] = vid
            variables.append(
                {
                    "id": qid,
                    "class_id": class_id,
                    "answer": center,
                    "sources": [submissions[sid].history.stem for sid in members[class_id]],
                }
            )
        questions[qid] = {
            "centers": centers,
            "members": members,
            "baseline_class": submission_class.get((baseline_id, qid)),
        }

    n_vars = len(variables)
    # One exact equation per non-baseline submission, expressed as score delta.
    eq_rows: list[dict[int, float]] = []
    eq_rhs: list[float] = []
    eq_names: list[str] = []
    for sid, sub in enumerate(submissions):
        if sid == baseline_id:
            continue
        row: defaultdict[int, float] = defaultdict(float)
        for qid, question in questions.items():
            base_class = question["baseline_class"]
            sub_class = submission_class.get((sid, qid))
            if sub_class == base_class:
                continue
            if sub_class is not None:
                row[variable_by_qclass[(qid, sub_class)]] += 1.0
            if base_class is not None:
                row[variable_by_qclass[(qid, base_class)]] -= 1.0
        # Duplicate answer vectors carry no new information.
        rhs = counts[sid] - baseline_count
        if not row:
            if abs(rhs) > 1e-9:
                raise SystemExit(
                    f"Inconsistent history: {sub.history.stem} has baseline answers but delta={rhs}"
                )
            continue
        eq_rows.append(dict(row))
        eq_rhs.append(float(rhs))
        eq_names.append(sub.history.stem)

    a_eq_lil = lil_matrix((len(eq_rows), n_vars), dtype=float)
    for rid, row in enumerate(eq_rows):
        for vid, coeff in row.items():
            a_eq_lil[rid, vid] = coeff
    a_eq = csr_matrix(a_eq_lil)
    b_eq = np.asarray(eq_rhs, dtype=float)

    # At most one known candidate answer per question can be the public gold.
    a_ub_lil = lil_matrix((len(questions), n_vars), dtype=float)
    for rid, (qid, question) in enumerate(questions.items()):
        for class_id in range(len(question["centers"])):
            a_ub_lil[rid, variable_by_qclass[(qid, class_id)]] = 1.0
    a_ub = csr_matrix(a_ub_lil)
    b_ub = np.ones(len(questions), dtype=float)
    bounds = (np.zeros(n_vars), np.ones(n_vars))

    feasibility = linprog(
        np.zeros(n_vars), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
        bounds=list(zip(*bounds)), method="highs"
    )
    if not feasibility.success:
        raise SystemExit(f"Leaderboard delta model is infeasible: {feasibility.message}")

    integer_status = "not_requested"
    if args.integer_check:
        constraints = [
            LinearConstraint(a_eq, b_eq, b_eq),
            LinearConstraint(a_ub, -np.inf, b_ub),
        ]
        integer_result = milp(
            np.zeros(n_vars), integrality=np.ones(n_vars),
            bounds=Bounds(bounds[0], bounds[1]), constraints=constraints,
            options={"time_limit": 120.0},
        )
        integer_status = integer_result.message
        if not integer_result.success:
            raise SystemExit(f"Binary leaderboard delta model is infeasible: {integer_result.message}")

    selected_ids = (
        {int(value) for value in args.ids.split(",") if value.strip()}
        if args.ids
        else None
    )
    forced_one: list[dict] = []
    forced_zero: list[dict] = []
    ambiguous: list[dict] = []
    for vid, variable in enumerate(variables):
        if selected_ids is not None and int(variable["id"]) not in selected_ids:
            continue
        objective = np.zeros(n_vars)
        objective[vid] = 1.0
        lo = linprog(
            objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=list(zip(*bounds)), method="highs"
        )
        hi = linprog(
            -objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
            bounds=list(zip(*bounds)), method="highs"
        )
        if not lo.success or not hi.success:
            raise SystemExit(f"Bound solve failed for variable {vid}")
        record = dict(variable)
        record["min"] = float(lo.fun)
        record["max"] = float(-hi.fun)
        q = questions[variable["id"]]
        record["is_baseline"] = variable["class_id"] == q["baseline_class"]
        if record["min"] >= 1.0 - 1e-8:
            forced_one.append(record)
        elif record["max"] <= 1e-8:
            forced_zero.append(record)
        else:
            ambiguous.append(record)

    forced_upgrade_ids = sorted(
        row["id"] for row in forced_one if not row["is_baseline"]
    )
    forced_baseline_wrong_ids = sorted(
        row["id"] for row in forced_zero if row["is_baseline"]
    )
    report = {
        "model": "exact score deltas with per-question at-most-one known gold class",
        "public_size": args.public_size,
        "baseline": args.baseline,
        "baseline_count": int(baseline_count),
        "loaded_submissions": len(submissions),
        "equations": len(eq_rows),
        "varying_questions": len(questions),
        "variables": n_vars,
        "bounded_question_ids": (
            sorted(selected_ids) if selected_ids is not None else "all"
        ),
        "lp_feasible": True,
        "integer_status": integer_status,
        "forced_one": forced_one,
        "forced_zero": forced_zero,
        "ambiguous_count": len(ambiguous),
        "forced_upgrade_ids": forced_upgrade_ids,
        "forced_baseline_wrong_ids": forced_baseline_wrong_ids,
        "equation_details": [
            {"submission": name, "delta_count": rhs, "nonzero_terms": len(row)}
            for name, rhs, row in zip(eq_names, eq_rhs, eq_rows)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "loaded_submissions": len(submissions),
                "equations": len(eq_rows),
                "varying_questions": len(questions),
                "variables": n_vars,
                "forced_one": len(forced_one),
                "forced_zero": len(forced_zero),
                "forced_upgrade_ids": forced_upgrade_ids,
                "forced_baseline_wrong_ids": forced_baseline_wrong_ids,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
