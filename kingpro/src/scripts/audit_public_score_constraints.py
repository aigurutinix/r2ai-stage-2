"""Audit answer classes against exact public-score constraints.

Every scored submission contributes one integer equation: the sum of correct
answer classes selected by that submission equals its public correct count.
For each question, at most one historical answer class may be gold.  This
script solves the resulting binary feasibility problem and samples diverse
feasible vertices with deterministic pseudo-random objectives.

The output is diagnostic only.  A class that never appears in the sampled
solutions is not proved impossible; a class fixed by explicit min/max MILPs is
an exact consequence of the locally recorded leaderboard equations.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from infer_public_ensemble import (
    ROOT,
    cluster_values,
    load_history,
    numeric_answer,
    read_submission,
    score_to_count,
)


def build_problem(history_path: Path, public_size: int, tolerance: float) -> dict[str, Any]:
    history = load_history(history_path)
    submissions = [sub for row in history if (sub := read_submission(ROOT, row)) is not None]
    submissions = [sub for sub in submissions if len(sub.entries) == 1012]
    submissions.sort(key=lambda sub: sub.history.submission_id)
    if len(submissions) < 2:
        raise SystemExit("Need at least two complete local scored submissions")

    qids = sorted(set().union(*(set(sub.entries) for sub in submissions)))
    variables: list[dict[str, Any]] = []
    variable_for_value: dict[tuple[int, float], int] = {}
    question_variables: dict[int, list[int]] = defaultdict(list)

    for qid in qids:
        values = [
            value
            for sub in submissions
            if (value := numeric_answer(sub.entries.get(qid, {}))) is not None
        ]
        centers, mapping = cluster_values(values, tolerance)
        for class_id, center in enumerate(centers):
            index = len(variables)
            variables.append({"id": qid, "class_id": class_id, "answer": center})
            question_variables[qid].append(index)
        for value, class_id in mapping.items():
            variable_for_value[(qid, value)] = question_variables[qid][class_id]

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    for sid, sub in enumerate(submissions):
        for qid, entry in sub.entries.items():
            value = numeric_answer(entry)
            if value is None:
                continue
            row_indices.append(sid)
            col_indices.append(variable_for_value[(qid, value)])
            data.append(1.0)
    score_matrix = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(submissions), len(variables)),
        dtype=float,
    ).tocsr()
    counts = np.asarray(
        [score_to_count(sub.history.answer_accuracy, public_size) for sub in submissions],
        dtype=float,
    )

    q_rows: list[int] = []
    q_cols: list[int] = []
    for row, qid in enumerate(qids):
        for column in question_variables[qid]:
            q_rows.append(row)
            q_cols.append(column)
    question_matrix = coo_matrix(
        (np.ones(len(q_rows)), (q_rows, q_cols)),
        shape=(len(qids), len(variables)),
        dtype=float,
    ).tocsr()
    matrix = vstack([score_matrix, question_matrix], format="csr")
    lower = np.r_[counts, np.zeros(len(qids))]
    upper = np.r_[counts, np.ones(len(qids))]
    constraint = LinearConstraint(matrix, lower, upper)
    return {
        "submissions": submissions,
        "variables": variables,
        "variable_for_value": variable_for_value,
        "question_variables": question_variables,
        "constraint": constraint,
        "counts": counts,
    }


def solve(c: np.ndarray, constraint: LinearConstraint, time_limit: float) -> Any:
    return milp(
        c=c,
        integrality=np.ones(len(c), dtype=np.uint8),
        bounds=Bounds(np.zeros(len(c)), np.ones(len(c))),
        constraints=constraint,
        options={"time_limit": time_limit},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "configs/leaderboard_history.csv")
    parser.add_argument("--public-size", type=int, default=506)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--baseline", default="sub_top123_candidate_v210_semantic_scope_batch2")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--exact-top", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    problem = build_problem(args.history, args.public_size, args.tolerance)
    submissions = problem["submissions"]
    variables = problem["variables"]
    constraint = problem["constraint"]
    variable_for_value = problem["variable_for_value"]
    baseline = next((sub for sub in submissions if sub.history.stem == args.baseline), None)
    if baseline is None:
        raise SystemExit(f"Baseline not present in score history: {args.baseline}")

    size = len(variables)
    rng = random.Random(args.seed)
    solutions: list[np.ndarray] = []
    statuses: list[dict[str, Any]] = []
    objectives = [np.zeros(size)]
    for _ in range(args.samples):
        objectives.append(np.asarray([rng.uniform(-1.0, 1.0) for _ in range(size)]))
    for index, objective in enumerate(objectives):
        fitted = solve(objective, constraint, args.time_limit)
        statuses.append(
            {
                "sample": index,
                "success": bool(fitted.success),
                "status": int(fitted.status),
                "message": str(fitted.message),
            }
        )
        if fitted.x is not None and np.max(np.abs(np.rint(fitted.x) - fitted.x)) < 1e-5:
            solutions.append(np.rint(fitted.x).astype(np.uint8))
    if not solutions:
        raise SystemExit("No integral feasible solution found")

    frequencies = np.mean(np.vstack(solutions), axis=0)
    baseline_rows: list[dict[str, Any]] = []
    for qid, entry in sorted(baseline.entries.items()):
        value = numeric_answer(entry)
        variable = variable_for_value.get((qid, value)) if value is not None else None
        if variable is None:
            continue
        alternatives = sorted(
            (
                {
                    "answer": variables[column]["answer"],
                    "sample_frequency": round(float(frequencies[column]), 6),
                    "variable": column,
                }
                for column in problem["question_variables"][qid]
                if column != variable
            ),
            key=lambda row: (-row["sample_frequency"], row["answer"]),
        )
        baseline_rows.append(
            {
                "id": qid,
                "question": entry.get("question", ""),
                "baseline_answer": value,
                "baseline_sample_frequency": round(float(frequencies[variable]), 6),
                "baseline_variable": variable,
                "best_alternative": alternatives[0] if alternatives else None,
            }
        )
    baseline_rows.sort(
        key=lambda row: (
            row["baseline_sample_frequency"],
            -float((row.get("best_alternative") or {}).get("sample_frequency", 0.0)),
            row["id"],
        )
    )

    exact_checks = []
    for row in baseline_rows[: args.exact_top]:
        columns = [row["baseline_variable"]]
        if row.get("best_alternative"):
            columns.append(int(row["best_alternative"]["variable"]))
        for column in columns:
            objective = np.zeros(size)
            objective[column] = -1.0
            maximum = solve(objective, constraint, args.time_limit)
            max_value = float(maximum.x[column]) if maximum.x is not None else None
            objective[column] = 1.0
            minimum = solve(objective, constraint, args.time_limit)
            min_value = float(minimum.x[column]) if minimum.x is not None else None
            exact_checks.append(
                {
                    "id": row["id"],
                    "answer": variables[column]["answer"],
                    "is_baseline": column == row["baseline_variable"],
                    "minimum": min_value,
                    "maximum": max_value,
                    "forced_zero": max_value is not None and max_value < 0.5,
                    "forced_one": min_value is not None and min_value > 0.5,
                    "min_status": int(minimum.status),
                    "max_status": int(maximum.status),
                }
            )

    report = {
        "schema_version": 1,
        "claim_limit": (
            "Leaderboard equations only. Sample frequencies are triage signals, not probabilities. "
            "Only forced_zero/forced_one from successful exact MILPs are logical consequences."
        ),
        "history": str(args.history.resolve()),
        "baseline": args.baseline,
        "public_size": args.public_size,
        "submission_count": len(submissions),
        "variable_count": size,
        "question_count": len(problem["question_variables"]),
        "feasible_solution_count": len(solutions),
        "sample_statuses": statuses,
        "baseline_ranking": baseline_rows,
        "exact_checks": exact_checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "submissions": len(submissions),
                "variables": size,
                "feasible_solutions": len(solutions),
                "exact_checks": len(exact_checks),
                "forced_zero": sum(bool(row["forced_zero"]) for row in exact_checks),
                "forced_one": sum(bool(row["forced_one"]) for row in exact_checks),
                "top_baseline_ids": [row["id"] for row in baseline_rows[:20]],
                "output": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
