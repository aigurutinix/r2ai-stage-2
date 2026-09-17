"""Recover answer-variant facts from aggregate leaderboard history.

Each scored submission gives an integer equation: exactly ``round(acc * 506)``
public questions were correct.  Local candidate directories reveal which
answer variant each equation selected.  A binary MILP can therefore prove
some variants always correct/incorrect across every feasible explanation of
the aggregate scores, without access to hidden question IDs or gold answers.

This is an offline forensic tool.  It never uploads, edits, or labels a
variant unless the constraint system forces that value in every solution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SIZE = 506
DEFAULT_REFERENCE = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
DEFAULT_OUTPUT = ROOT / "build" / "v257_leaderboard_answer_constraints.json"
AUTHORITATIVE_VERSION_DIRS = {
    "v217": "sub_top123_candidate_v217_missing_panel_operand_batch3",
    "v224": "sub_top123_candidate_v224_hut_parent_inventory_scope_batch8",
    "v225": "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9",
    "v226": "sub_top123_candidate_v226_scope_pair_public_ablation",
    "v265": "sub_top123_candidate_v265_q24_retrieval_crosscheck_ablation",
    "v269": "sub_top123_candidate_v269_source_lineage_control",
    "v276": "sub_v276_q638_fix",
    "v287": "sub_v287_scope3",
    "v290": "sub_v290_scope2",
    "v297": "sub_v297_scope2",
}
AUTHORITATIVE_SUBMISSION_DIRS = {
    3622: "sub_top123_candidate_v207_semantic_batch6_final",
    3668: "sub_top123_candidate_v208_semantic_batch2",
    3674: "sub_top123_candidate_v209_q15_board_role_r2",
    3685: "sub_top123_candidate_v210_semantic_scope_batch2",
    3696: "sub_top123_candidate_v217_missing_panel_operand_batch3",
    3699: "sub_top123_candidate_v218_existing_table_completeness_batch4",
    3722: "sub_top123_candidate_v224_hut_parent_inventory_scope_batch8",
    3723: "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9",
    3726: "sub_top123_candidate_v226_scope_pair_public_ablation",
    3740: "sub_top123_candidate_v265_q24_retrieval_crosscheck_ablation",
    3741: "sub_top123_candidate_v269_source_lineage_control",
    3742: "sub_v276_q638_fix",
    3744: "sub_v287_scope3",
    3745: "sub_v290_scope2",
    3747: "sub_v297_scope2",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def answer_key(value: Any) -> str:
    """Cluster submitted two-decimal answers by scorer-equivalent value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"text:{value!s}"
    if not math.isfinite(number):
        return f"nonfinite:{number}"
    rounded = round(number, 2)
    if rounded == 0:
        rounded = 0.0
    return f"num:{rounded:.2f}"


def candidate_answers(path: Path) -> dict[int, str]:
    rows = load_json(path / "submission.json")
    answers = {int(row["id"]): answer_key(row.get("answer")) for row in rows}
    if len(rows) != 1012 or len(answers) != 1012:
        raise ValueError(f"{path}: expected 1012 unique answers")
    return answers


def local_candidates(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("sub_")
        and (path / "submission.json").is_file()
    )


def match_candidate(filename: str, candidates: list[Path]) -> tuple[Path | None, str]:
    stem = Path(filename).stem
    exact = [path for path in candidates if path.name == stem]
    if len(exact) == 1:
        return exact[0], "exact"
    prefix = [path for path in candidates if path.name.startswith(stem)]
    if len(prefix) == 1:
        return prefix[0], "unique_prefix"
    version = re.search(r"(?:^|_)v(\d+)(?:_|$)", stem)
    if version:
        same_version = [
            path
            for path in candidates
            if re.search(rf"(?:^|_)v{version.group(1)}(?:_|$)", path.name)
        ]
        if len(same_version) == 1:
            return same_version[0], "unique_version"
    return None, "ambiguous" if prefix or version else "unmatched"


def collect_score_rows(score_dir: Path) -> tuple[list[dict], list[dict]]:
    by_id: dict[int, dict] = {}
    for path in sorted(score_dir.glob("leaderboard_scores*.json")):
        try:
            payload = load_json(path)
        except (OSError, ValueError, TypeError):
            continue
        for row in payload.get("submissions", []):
            accuracy = (row.get("scores") or {}).get("ANSWER_ACCURACY")
            if accuracy is None:
                continue
            item = dict(row)
            item["score_source"] = str(path.relative_to(ROOT))
            by_id[int(row["id"])] = item

    state_path = ROOT / "knowledge/vothuong/current_state.json"
    if state_path.is_file():
        state = load_json(state_path)
        public = state.get("authoritative_public_state", {})
        for name in ("selected_champion", "latest_submission"):
            item = public.get(name, {})
            if not item:
                continue
            version = str(item.get("version", ""))
            sid = int(item["submission_id"])
            accuracy = item.get("scores", {}).get("answer_accuracy")
            # A transient Running/null state must not erase a newer finished
            # score already collected from the authenticated API snapshot.
            if accuracy is None and sid in by_id:
                continue
            by_id[sid] = {
                "id": sid,
                "filename": item.get("filename") or f"sub_top123_candidate_{version}.zip",
                "scores": {"ANSWER_ACCURACY": accuracy},
                "version_hint": version,
                "score_source": str(state_path.relative_to(ROOT)),
            }
        for item in state.get("measured_experiments", []):
            sid = int(item["submission_id"])
            by_id[sid] = {
                "id": sid,
                "filename": f"sub_top123_candidate_{item['version']}.zip",
                "scores": {"ANSWER_ACCURACY": item.get("answer_accuracy")},
                "version_hint": str(item["version"]),
                "score_source": str(state_path.relative_to(ROOT)),
            }
    valid = [row for row in by_id.values() if (row.get("scores") or {}).get("ANSWER_ACCURACY") is not None]
    return sorted(valid, key=lambda row: int(row["id"])), []


@dataclass
class Observation:
    submission_id: int
    candidate: str
    candidate_sha256: str
    answer_count: int
    answers: dict[int, str]
    score_source: str
    match_kind: str


def build_observations(root: Path, score_dir: Path) -> tuple[list[Observation], list[dict]]:
    candidates = local_candidates(root)
    score_rows, _ = collect_score_rows(score_dir)
    observations: list[Observation] = []
    skipped: list[dict] = []
    for row in score_rows:
        submission_id = int(row["id"])
        version_hint = str(row.get("version_hint", ""))
        authoritative = root / AUTHORITATIVE_VERSION_DIRS.get(version_hint, "")
        authoritative_id = root / AUTHORITATIVE_SUBMISSION_DIRS.get(submission_id, "")
        if submission_id in AUTHORITATIVE_SUBMISSION_DIRS and (authoritative_id / "submission.json").is_file():
            path, match_kind = authoritative_id, "authoritative_submission_mapping"
        elif version_hint in AUTHORITATIVE_VERSION_DIRS and (authoritative / "submission.json").is_file():
            path, match_kind = authoritative, "authoritative_state_mapping"
        else:
            path, match_kind = match_candidate(str(row.get("filename", "")), candidates)
        if path is None and row.get("version_hint"):
            version = str(row["version_hint"])
            matches = [p for p in candidates if re.search(rf"(?:^|_){re.escape(version)}(?:_|$)", p.name)]
            if len(matches) == 1:
                path, match_kind = matches[0], "state_unique_version"
        if path is None:
            skipped.append(
                {
                    "submission_id": int(row["id"]),
                    "filename": row.get("filename"),
                    "reason": match_kind,
                }
            )
            continue
        accuracy = float(row["scores"]["ANSWER_ACCURACY"])
        count = int(round(accuracy * PUBLIC_SIZE))
        if abs(accuracy - count / PUBLIC_SIZE) > 0.00006:
            skipped.append(
                {
                    "submission_id": int(row["id"]),
                    "filename": row.get("filename"),
                    "reason": "score_not_consistent_with_506_after_display_rounding",
                }
            )
            continue
        try:
            answers = candidate_answers(path)
        except (OSError, ValueError, TypeError) as exc:
            skipped.append(
                {"submission_id": int(row["id"]), "filename": row.get("filename"), "reason": str(exc)}
            )
            continue
        observations.append(
            Observation(
                submission_id=int(row["id"]),
                candidate=path.name,
                candidate_sha256=sha256_file(path / "submission.json"),
                answer_count=count,
                answers=answers,
                score_source=str(row.get("score_source", "")),
                match_kind=match_kind,
            )
        )

    # Multiple submissions of byte-identical answer vectors add no equation.
    # Conflicting scores for one vector invalidate that mapping and are dropped.
    grouped: dict[tuple[tuple[int, str], ...], list[Observation]] = {}
    for item in observations:
        signature = tuple(sorted(item.answers.items()))
        grouped.setdefault(signature, []).append(item)
    deduped: list[Observation] = []
    for group in grouped.values():
        counts = {item.answer_count for item in group}
        if len(counts) != 1:
            skipped.extend(
                {
                    "submission_id": item.submission_id,
                    "filename": item.candidate,
                    "reason": f"identical_answer_vector_conflicting_counts:{sorted(counts)}",
                }
                for item in group
            )
            continue
        deduped.append(max(group, key=lambda item: item.submission_id))
    return sorted(deduped, key=lambda item: item.submission_id), skipped


def solve_constraints(observations: list[Observation], reference: dict[int, str]) -> dict:
    if len(observations) < 2:
        raise ValueError("need at least two distinct scored answer vectors")
    variants: dict[int, set[str]] = {}
    for qid in reference:
        values = {item.answers[qid] for item in observations}
        if len(values) > 1:
            variants[qid] = values
    keys = [(qid, value) for qid in sorted(variants) for value in sorted(variants[qid])]
    index = {key: pos for pos, key in enumerate(keys)}
    intercept = len(keys)
    nvars = intercept + 1

    eq_rows = []
    eq_values = []
    for item in observations:
        row = np.zeros(nvars)
        for qid in variants:
            row[index[(qid, item.answers[qid])]] = 1.0
        row[intercept] = 1.0
        eq_rows.append(row)
        eq_values.append(float(item.answer_count))

    constraints: list[LinearConstraint] = [
        LinearConstraint(csr_matrix(np.asarray(eq_rows)), np.asarray(eq_values), np.asarray(eq_values))
    ]
    for qid, values in variants.items():
        row = np.zeros(nvars)
        for value in values:
            row[index[(qid, value)]] = 1.0
        constraints.append(LinearConstraint(csr_matrix(row.reshape(1, -1)), -np.inf, 1.0))

    lower = np.zeros(nvars)
    upper = np.ones(nvars)
    upper[intercept] = PUBLIC_SIZE
    bounds = Bounds(lower, upper)
    integrality = np.ones(nvars)
    base = milp(
        np.zeros(nvars),
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": 30},
    )
    if not base.success:
        raise ValueError(f"leaderboard constraints infeasible: {base.message}")

    # Fast exact propagation over pairwise score differences.  For a changed
    # question, the contribution is x(new)-x(old) in {-1,0,1}.  If a pair's
    # required delta equals the sum of all per-question maxima/minima, every
    # term is forced to that extreme.  This is strictly weaker than solving
    # 2*N MILPs but finishes in seconds and never makes a non-forced claim.
    status: dict[tuple[int, str], int] = {}

    def assign(key: tuple[int, str], value: int) -> bool:
        previous = status.get(key)
        if previous is not None:
            if previous != value:
                raise ValueError(f"contradictory propagated status for {key}")
            return False
        status[key] = value
        changed = True
        if value == 1:
            qid, _ = key
            for sibling in variants[qid]:
                if sibling != key[1]:
                    assign((qid, sibling), 0)
        return changed

    pairs = []
    for left_index, left in enumerate(observations):
        for right in observations[left_index + 1 :]:
            changed_qids = [
                qid for qid in variants if left.answers[qid] != right.answers[qid]
            ]
            if not changed_qids:
                continue
            pairs.append((left, right, changed_qids, right.answer_count - left.answer_count))

    progress = True
    while progress:
        progress = False
        for left, right, changed_qids, required in pairs:
            terms = []
            total_min = 0
            total_max = 0
            for qid in changed_qids:
                old = (qid, left.answers[qid])
                new = (qid, right.answers[qid])
                old_value = status.get(old)
                new_value = status.get(new)
                if old_value == 1:
                    assign(new, 0)
                    new_value = 0
                if new_value == 1:
                    assign(old, 0)
                    old_value = 0
                if old_value is not None and new_value is not None:
                    choices = {new_value - old_value}
                elif old_value == 0:
                    choices = {0, 1}
                elif new_value == 0:
                    choices = {-1, 0}
                else:
                    choices = {-1, 0, 1}
                lo, hi = min(choices), max(choices)
                total_min += lo
                total_max += hi
                terms.append((old, new, choices))
            if required < total_min or required > total_max:
                raise ValueError(
                    f"pair constraint infeasible: {left.submission_id}->{right.submission_id} "
                    f"requires {required}, range {total_min}..{total_max}"
                )
            fixed_sum = sum(next(iter(choices)) for _, _, choices in terms if len(choices) == 1)
            flexible = [(old, new, choices) for old, new, choices in terms if len(choices) > 1]
            if len(flexible) == 1:
                old, new, choices = flexible[0]
                needed = required - fixed_sum
                if needed not in choices:
                    raise ValueError("single remaining pair term cannot satisfy score delta")
                if needed == 1:
                    progress |= assign(old, 0)
                    progress |= assign(new, 1)
                elif needed == -1:
                    progress |= assign(old, 1)
                    progress |= assign(new, 0)
                elif needed == 0:
                    # Differing variants cannot both be correct under the
                    # per-question <=1 constraint. Zero contribution therefore
                    # forces both to zero (private question or a third gold value).
                    progress |= assign(old, 0)
                    progress |= assign(new, 0)
            if required == total_max:
                for old, new, choices in terms:
                    if max(choices) == 1:
                        progress |= assign(old, 0)
                        progress |= assign(new, 1)
                    elif choices == {-1}:
                        progress |= assign(old, 1)
                        progress |= assign(new, 0)
            elif required == total_min:
                for old, new, choices in terms:
                    if min(choices) == -1:
                        progress |= assign(old, 1)
                        progress |= assign(new, 0)
                    elif choices == {1}:
                        progress |= assign(old, 0)
                        progress |= assign(new, 1)

    forced: list[dict] = [
        {
            "id": qid,
            "answer_variant": value,
            "forced_correct": bool(forced_value),
            "is_reference_variant": reference[qid] == value,
        }
        for (qid, value), forced_value in sorted(status.items())
    ]
    small_pairs = [
        {
            "left_submission_id": left.submission_id,
            "left_candidate": left.candidate,
            "right_submission_id": right.submission_id,
            "right_candidate": right.candidate,
            "answer_count_delta": required,
            "changed_question_ids": changed_qids,
            "changes": [
                {
                    "id": qid,
                    "left_answer": left.answers[qid],
                    "right_answer": right.answers[qid],
                }
                for qid in changed_qids
            ],
        }
        for left, right, changed_qids, required in pairs
        if len(changed_qids) <= 12
    ]
    small_pairs.sort(
        key=lambda item: (
            len(item["changed_question_ids"]),
            item["left_submission_id"],
            item["right_submission_id"],
        )
    )

    by_qid: dict[int, list[dict]] = {}
    for item in forced:
        by_qid.setdefault(int(item["id"]), []).append(item)
    forced_repairs = []
    for qid, items in sorted(by_qid.items()):
        reference_item = next((item for item in items if item["is_reference_variant"]), None)
        correct = [item for item in items if item["forced_correct"]]
        if reference_item and not reference_item["forced_correct"] and len(correct) == 1:
            forced_repairs.append(
                {
                    "id": qid,
                    "reference_answer": reference[qid],
                    "forced_answer": correct[0]["answer_variant"],
                }
            )
    return {
        "changed_question_count": len(variants),
        "variant_variable_count": len(keys),
        "equation_count": len(observations),
        "pair_constraint_count": len(pairs),
        "small_pair_constraints": small_pairs,
        "feasible_intercept": int(round(base.x[intercept])),
        "forced_variant_count": len(forced),
        "forced_variants": forced,
        "forced_reference_repairs": forced_repairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--score-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    reference_path = args.reference.resolve()
    observations, skipped = build_observations(root, args.score_dir.resolve())
    reference = candidate_answers(reference_path)
    solution = solve_constraints(observations, reference)
    payload = {
        "schema_version": 1,
        "kind": "aggregate_leaderboard_answer_constraint_solver",
        "public_size": PUBLIC_SIZE,
        "reference_candidate": reference_path.name,
        "reference_submission_sha256": sha256_file(reference_path / "submission.json"),
        "observation_count": len(observations),
        "skipped_count": len(skipped),
        "policy": {
            "hidden_gold_access": False,
            "automatic_mutation": False,
            "forced_means": "same binary value in every feasible solution",
            "source_adjudication_required_before_candidate_build": True,
        },
        "observations": [
            {
                "submission_id": item.submission_id,
                "candidate": item.candidate,
                "candidate_sha256": item.candidate_sha256,
                "answer_count": item.answer_count,
                "score_source": item.score_source,
                "match_kind": item.match_kind,
            }
            for item in observations
        ],
        "skipped": skipped,
        **solution,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "observation_count",
                    "skipped_count",
                    "changed_question_count",
                    "variant_variable_count",
                    "equation_count",
                    "forced_variant_count",
                    "forced_reference_repairs",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
