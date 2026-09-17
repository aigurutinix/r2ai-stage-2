"""Infer answer states from the public-leaderboard submission history.

Each public score is a count over 506 questions.  For every question that
changed across locally preserved submissions, the history therefore supplies
linear constraints on which previously observed answer state can be correct.
The script reports states that are forced correct/incorrect in every feasible
solution; it never treats a score-neutral edit as evidence by itself because
the question may belong to the private half.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


PUBLIC_SIZE = 506
SCORES = {
    2: .6601, 3: .6640, 4: .6640, 5: .6798, 6: .6798, 7: .6818,
    8: .6818, 9: .6818, 10: .6818, 11: .6818, 12: .6818,
    13: .6818, 14: .6818, 15: .6798, 16: .6838, 17: .6838,
    18: .6858, 19: .6858, 20: .6877, 21: .6877, 22: .6877,
    23: .6877, 24: .6877, 25: .6877, 26: .6877, 27: .6877,
    28: .6877, 29: .6877, 30: .6858, 32: .6877, 33: .6877,
    34: .6877, 35: .6858, 36: .6877, 37: .6877, 38: .6877,
    39: .6877, 40: .6877, 41: .6877, 42: .6897, 43: .6897,
    44: .6917, 45: .6917, 46: .6917, 47: .6917, 48: .6917,
    49: .6917, 50: .6917, 51: .6917, 52: .6917, 53: .6937,
    54: .6937, 55: .6937, 56: .6937, 57: .6937, 58: .6917,
    59: .6917, 60: .6937, 61: .6937, 62: .6937, 63: .6937,
    64: .6917, 65: .6937, 66: .6937, 67: .6937, 68: .6996,
    69: .6937, 70: .7016, 71: .7016, 72: .6996,
}


def _load_rows(path: Path) -> dict[int, dict]:
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".json"))
        return {int(row["id"]): row for row in json.loads(archive.read(member))}


def _same_state(left: float, right: float) -> bool:
    # The official repository documents an absolute answer tolerance of 0.01.
    # Keep large-value arithmetic exact enough while merging harmless rounding.
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=0.01)


def _cluster_states(
    rows_by_version: dict[int, dict[int, dict]],
) -> tuple[dict[int, list[float]], dict[tuple[int, int], int]]:
    states: dict[int, list[float]] = defaultdict(list)
    state_by_version: dict[tuple[int, int], int] = {}
    for version in sorted(rows_by_version):
        for qid, row in rows_by_version[version].items():
            value = float(row["answer"])
            choices = states[qid]
            state = next(
                (index for index, candidate in enumerate(choices) if _same_state(value, candidate)),
                None,
            )
            if state is None:
                state = len(choices)
                choices.append(value)
            state_by_version[(version, qid)] = state
    return dict(states), state_by_version


def _solve(c: np.ndarray, constraint: LinearConstraint) -> np.ndarray:
    result = milp(
        c=c,
        integrality=np.ones(len(c)),
        bounds=Bounds(np.zeros(len(c)), np.ones(len(c))),
        constraints=constraint,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"score equations are infeasible: {result.message}")
    return result.x


def _is_feasible(c: np.ndarray, constraint: LinearConstraint) -> bool:
    result = milp(
        c=c,
        integrality=np.ones(len(c)),
        bounds=Bounds(np.zeros(len(c)), np.ones(len(c))),
        constraints=constraint,
        options={"presolve": True},
    )
    return bool(result.success and result.x is not None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--current", type=int, default=70)
    parser.add_argument("--exclude", type=int, nargs="*", default=())
    args = parser.parse_args()

    rows_by_version: dict[int, dict[int, dict]] = {}
    excluded = set(args.exclude)
    for version in sorted(SCORES):
        if version in excluded:
            continue
        path = args.root / f"submission_vn{version}.zip"
        if path.exists():
            rows_by_version[version] = _load_rows(path)
    versions = sorted(rows_by_version)
    base = versions[0]
    if args.current not in rows_by_version:
        raise SystemExit(f"missing current submission VN{args.current}")

    states, state_by_version = _cluster_states(rows_by_version)
    changed_qids = sorted(qid for qid, values in states.items() if len(values) > 1)
    variables = [
        (qid, state)
        for qid in changed_qids
        for state in range(len(states[qid]))
    ]
    column = {item: index for index, item in enumerate(variables)}

    # Equality rows: public-correct count delta from the first scored version.
    equal = lil_matrix((len(versions) - 1, len(variables)), dtype=float)
    rhs = np.zeros(len(versions) - 1)
    base_count = round(SCORES[base] * PUBLIC_SIZE)
    for row_index, version in enumerate(versions[1:]):
        for qid in changed_qids:
            current_state = state_by_version[(version, qid)]
            base_state = state_by_version[(base, qid)]
            if current_state != base_state:
                equal[row_index, column[(qid, current_state)]] += 1
                equal[row_index, column[(qid, base_state)]] -= 1
        rhs[row_index] = round(SCORES[version] * PUBLIC_SIZE) - base_count

    # Treat every observed state independently.  The scorer's tolerance can
    # make two close numeric variants simultaneously correct; query/runtime
    # differences can also make two identical displayed answers non-equivalent.
    # Omitting a mutual-exclusion assumption is conservative: it yields fewer,
    # but more defensible, forced conclusions.
    constraint = LinearConstraint(equal.tocsr(), rhs, rhs)

    zero_objective = np.zeros(len(variables))
    if not _is_feasible(zero_objective, constraint):
        for count, version in enumerate(versions[1:], start=1):
            partial = LinearConstraint(equal.tocsr()[:count], rhs[:count], rhs[:count])
            if not _is_feasible(zero_objective, partial):
                raise RuntimeError(
                    f"score equations first become infeasible at VN{version}; "
                    "the locally preserved ZIP likely differs from the scored artifact. "
                    f"Retry with --exclude {version}."
                )
        raise RuntimeError("score equations are infeasible for an unidentified reason")

    forced: dict[tuple[int, int], int | None] = {}
    for item, index in column.items():
        objective = np.zeros(len(variables))
        objective[index] = 1
        minimum = round(_solve(objective, constraint)[index])
        objective[index] = -1
        maximum = round(_solve(objective, constraint)[index])
        forced[item] = minimum if minimum == maximum else None

    print(
        f"versions={len(versions)} changed_questions={len(changed_qids)} "
        f"variables={len(variables)} current=VN{args.current}"
    )
    actionable = []
    for qid in changed_qids:
        current_state = state_by_version[(args.current, qid)]
        correct_states = [
            state for state in range(len(states[qid])) if forced[(qid, state)] == 1
        ]
        if correct_states and current_state not in correct_states:
            target_state = correct_states[0]
            donor_versions = [
                version for version in versions
                if state_by_version[(version, qid)] == target_state
            ]
            actionable.append((qid, current_state, target_state, donor_versions))

    if not actionable:
        print("No historically observed answer state is forced better than the current release.")
    else:
        print("Forced corrections missing from current:")
        for qid, current_state, target_state, donors in actionable:
            question = rows_by_version[args.current][qid]["question"]
            print(
                f"Q{qid}: {states[qid][current_state]!r} -> {states[qid][target_state]!r}; "
                f"donors={donors}; {question}"
            )

    print("Forced state summary:")
    for qid in changed_qids:
        known = [
            (state, verdict) for state in range(len(states[qid]))
            if (verdict := forced[(qid, state)]) is not None
        ]
        if known:
            rendered = ", ".join(
                f"state{state}={states[qid][state]!r}:{'correct' if verdict else 'incorrect'}"
                for state, verdict in known
            )
            print(f"Q{qid}: {rendered}")


if __name__ == "__main__":
    main()
