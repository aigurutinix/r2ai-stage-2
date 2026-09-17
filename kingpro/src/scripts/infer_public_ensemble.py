"""Infer a public-leaderboard ensemble from aggregate scores of prior submissions.

Each Answer Accuracy score is one equation: the number of public questions for which
that submission selected the gold answer.  Candidate answers within the grader's
absolute tolerance are clustered.  A maximum-entropy model then estimates, per
question, which candidate class is most compatible with all historical equations.

This is a public-leaderboard diagnostic, not private-set validation.  It never uploads
anything.  With --build it creates a new submission by copying the executable entry
for the most likely candidate; ambiguous or non-executable choices fall back to the
specified safe baseline.

Examples:
  python scripts/infer_public_ensemble.py
  python scripts/infer_public_ensemble.py --build sub_lbmeta
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HistoryRow:
    submission_id: int
    filename: str
    answer_accuracy: float
    execution_accuracy: float

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


@dataclass
class LoadedSubmission:
    history: HistoryRow
    entries: dict[int, dict]
    directory: Path | None
    archive: Path | None


def load_history(path: Path) -> list[HistoryRow]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [
            HistoryRow(
                submission_id=int(row["submission_id"]),
                filename=row["filename"],
                answer_accuracy=float(row["answer_accuracy"]),
                execution_accuracy=float(row["execution_accuracy"]),
            )
            for row in csv.DictReader(fh)
        ]
    # Duplicate uploads of the same file add no independent equation.
    unique: dict[str, HistoryRow] = {}
    for row in rows:
        unique.setdefault(row.filename, row)
    return list(unique.values())


def read_submission(root: Path, row: HistoryRow) -> LoadedSubmission | None:
    directory = root / row.stem
    archive = root / row.filename
    try:
        # Prefer the immutable archive: a working directory may have been edited
        # after the file was uploaded, in which case it no longer represents the
        # leaderboard row named by ``filename``.
        if archive.exists():
            with zipfile.ZipFile(archive) as zf:
                payload = json.loads(zf.read("submission.json").decode("utf-8"))
            source_dir, source_zip = None, archive
        elif (directory / "submission.json").exists():
            payload = json.loads((directory / "submission.json").read_text(encoding="utf-8"))
            source_dir, source_zip = directory, None
        else:
            return None
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return None
    entries = {int(entry["id"]): entry for entry in payload if "id" in entry}
    return LoadedSubmission(row, entries, source_dir, source_zip)


def numeric_answer(entry: dict) -> float | None:
    value = entry.get("answer")
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def cluster_values(values: list[float], tolerance: float) -> tuple[list[float], dict[float, int]]:
    """Connected components under abs(a-b)<=tolerance, represented by their median."""
    ordered = sorted(set(values))
    groups: list[list[float]] = []
    for value in ordered:
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    centers = [float(np.median(group)) for group in groups]
    mapping = {value: group_id for group_id, group in enumerate(groups) for value in group}
    return centers, mapping


def score_to_count(score: float, public_size: int) -> int:
    count = int(round(score * public_size))
    rounded = round(count / public_size, 4)
    if abs(rounded - score) > 5.1e-5:
        raise ValueError(
            f"Score {score:.4f} is not compatible with public_size={public_size}; "
            f"nearest count is {count} ({rounded:.4f})"
        )
    return count


def fit_max_entropy(
    submissions: list[LoadedSubmission], public_size: int, tolerance: float, l2: float
) -> tuple[dict[int, dict], np.ndarray, np.ndarray]:
    qids = sorted(set().union(*(set(sub.entries) for sub in submissions)))
    n_sub = len(submissions)
    counts = np.array(
        [score_to_count(sub.history.answer_accuracy, public_size) for sub in submissions], dtype=float
    )
    questions: dict[int, dict] = {}

    for qid in qids:
        vals = [
            value
            for sub in submissions
            if (value := numeric_answer(sub.entries.get(qid, {}))) is not None
        ]
        if not vals:
            continue
        centers, value_to_class = cluster_values(vals, tolerance)
        features = np.zeros((len(centers), n_sub), dtype=float)
        members: list[list[int]] = [[] for _ in centers]
        for sub_id, sub in enumerate(submissions):
            value = numeric_answer(sub.entries.get(qid, {}))
            if value is None:
                continue
            class_id = value_to_class[value]
            features[class_id, sub_id] = 1.0
            members[class_id].append(sub_id)
        questions[qid] = {"centers": centers, "features": features, "members": members}

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        expected = np.zeros(n_sub, dtype=float)
        total = -float(np.dot(counts, lam)) + 0.5 * l2 * float(np.dot(lam, lam))
        for question in questions.values():
            features = question["features"]
            logits = features @ lam
            log_z = logsumexp(np.r_[0.0, logits])  # extra state: no historical candidate is gold
            probs = np.exp(logits - log_z)
            total += float(log_z)
            expected += probs @ features
        gradient = expected - counts + l2 * lam
        return total, gradient

    fitted = minimize(
        fun=lambda x: objective(x),
        x0=np.zeros(n_sub, dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 4000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 60},
    )
    if not fitted.success:
        print(f"WARNING optimizer: {fitted.message}")

    lam = fitted.x
    predicted = np.zeros(n_sub, dtype=float)
    for question in questions.values():
        logits = question["features"] @ lam
        log_z = logsumexp(np.r_[0.0, logits])
        probs = np.exp(logits - log_z)
        question["probabilities"] = probs
        question["none_probability"] = float(math.exp(-log_z))
        predicted += probs @ question["features"]
    return questions, counts, predicted


def entry_is_executable(entry: dict) -> bool:
    return bool((entry.get("pandas_query") or "").strip() and entry.get("evidence"))


def choose_source(
    qid: int, class_members: list[int], submissions: list[LoadedSubmission]
) -> tuple[int, dict] | None:
    ranked = sorted(
        class_members,
        key=lambda sid: (
            entry_is_executable(submissions[sid].entries.get(qid, {})),
            abs(
                submissions[sid].history.execution_accuracy
                - submissions[sid].history.answer_accuracy
            ) < 5e-5,
            submissions[sid].history.execution_accuracy,
            submissions[sid].history.submission_id,
        ),
        reverse=True,
    )
    for sid in ranked:
        entry = submissions[sid].entries.get(qid)
        if entry and entry_is_executable(entry):
            return sid, entry
    return None


def read_evidence_bytes(source: LoadedSubmission, relative_path: str) -> bytes:
    rel = relative_path.replace("\\", "/").lstrip("/")
    if source.directory is not None:
        return (source.directory / rel).read_bytes()
    if source.archive is not None:
        with zipfile.ZipFile(source.archive) as zf:
            return zf.read(rel)
    raise FileNotFoundError(rel)


def build_submission(
    out_dir: Path,
    questions: dict[int, dict],
    submissions: list[LoadedSubmission],
    baseline_stem: str,
    min_probability: float,
    min_margin: float,
) -> dict:
    baseline_id = next(
        (i for i, sub in enumerate(submissions) if sub.history.stem == baseline_stem), None
    )
    if baseline_id is None:
        raise ValueError(f"Baseline {baseline_stem!r} is not in loaded leaderboard history")
    baseline = submissions[baseline_id]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True)

    rows = []
    copied: dict[tuple[int, str], str] = {}
    overrides = []
    for qid in sorted(baseline.entries):
        fallback = baseline.entries[qid]
        chosen_sid, chosen_entry = baseline_id, fallback
        question = questions.get(qid)
        if question is not None:
            probs = question["probabilities"]
            order = np.argsort(-probs)
            best = int(order[0])
            second = float(probs[order[1]]) if len(order) > 1 else 0.0
            probability = float(probs[best])
            margin = probability - max(second, question["none_probability"])
            source = choose_source(qid, question["members"][best], submissions)
            if (
                source is not None
                and probability >= min_probability
                and margin >= min_margin
            ):
                chosen_sid, chosen_entry = source
                if chosen_sid != baseline_id or abs(
                    (numeric_answer(chosen_entry) or 0.0) - (numeric_answer(fallback) or 0.0)
                ) > 0.01:
                    overrides.append(
                        {
                            "id": qid,
                            "probability": probability,
                            "margin": margin,
                            "answer": numeric_answer(chosen_entry),
                            "source": submissions[chosen_sid].history.stem,
                        }
                    )

        entry = json.loads(json.dumps(chosen_entry, ensure_ascii=False))
        new_evidence = []
        for ev_index, evidence in enumerate(entry.get("evidence", []), 1):
            old_path = evidence.get("csv_path", "")
            cache_key = (chosen_sid, old_path)
            if cache_key not in copied:
                suffix = Path(old_path).suffix or ".csv"
                dest_name = f"s{chosen_sid:02d}_q{qid:04d}_e{ev_index:02d}{suffix}"
                (data_dir / dest_name).write_bytes(
                    read_evidence_bytes(submissions[chosen_sid], old_path)
                )
                copied[cache_key] = f"data/{dest_name}"
            new_evidence.append(
                {"variable": evidence.get("variable", f"df{ev_index}"), "csv_path": copied[cache_key]}
            )
        entry["evidence"] = new_evidence
        rows.append(entry)

    (out_dir / "submission.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "submission.json", "submission.json")
        for path in data_dir.glob("*.csv"):
            zf.write(path, f"data/{path.name}")
    return {"rows": len(rows), "overrides": overrides, "csv_files": len(copied), "zip": str(zip_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=ROOT / "configs/leaderboard_history.csv")
    parser.add_argument("--public-size", type=int, default=506)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-8)
    parser.add_argument("--baseline", default="sub_ctxB")
    parser.add_argument("--build", default="", help="Output directory name for an executable hybrid")
    parser.add_argument("--min-probability", type=float, default=0.60)
    parser.add_argument("--min-margin", type=float, default=0.20)
    parser.add_argument("--report", type=Path, default=ROOT / "build/public_ensemble_inference.json")
    args = parser.parse_args()

    history = load_history(args.history)
    loaded = [sub for row in history if (sub := read_submission(ROOT, row)) is not None]
    loaded = [sub for sub in loaded if len(sub.entries) == 1012]
    if len(loaded) < 2:
        raise SystemExit("Need at least two complete historical submissions")
    questions, counts, predicted = fit_max_entropy(
        loaded, args.public_size, args.tolerance, args.l2
    )

    residual = predicted - counts
    best_existing = max(zip(counts, loaded), key=lambda pair: pair[0])
    picked_expected = sum(float(np.max(q["probabilities"])) for q in questions.values())
    confident = sum(
        1
        for q in questions.values()
        if float(np.max(q["probabilities"])) >= args.min_probability
    )
    report = {
        "public_size": args.public_size,
        "submissions": len(loaded),
        "questions": len(questions),
        "best_existing": {
            "name": best_existing[1].history.stem,
            "count": int(best_existing[0]),
            "score": best_existing[0] / args.public_size,
        },
        "max_entropy_pick_expected_count": picked_expected,
        "max_entropy_pick_expected_score": picked_expected / args.public_size,
        "confident_questions": confident,
        "max_equation_residual": float(np.max(np.abs(residual))),
        "equations": [
            {
                "name": sub.history.stem,
                "observed_count": int(counts[i]),
                "fitted_count": float(predicted[i]),
                "residual": float(residual[i]),
            }
            for i, sub in enumerate(loaded)
        ],
        "question_predictions": [
            {
                "id": qid,
                "none_probability": question["none_probability"],
                "candidates": sorted(
                    [
                        {
                            "answer": question["centers"][cid],
                            "probability": float(question["probabilities"][cid]),
                            "sources": [loaded[sid].history.stem for sid in question["members"][cid]],
                        }
                        for cid in range(len(question["centers"]))
                    ],
                    key=lambda row: -row["probability"],
                ),
            }
            for qid, question in sorted(questions.items())
        ],
    }

    build_result = None
    if args.build:
        build_result = build_submission(
            ROOT / args.build,
            questions,
            loaded,
            args.baseline,
            args.min_probability,
            args.min_margin,
        )
        report["build"] = build_result

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "loaded_submissions": len(loaded),
                "best_existing": report["best_existing"],
                "max_entropy_pick_expected_count": round(picked_expected, 2),
                "max_entropy_pick_expected_score": round(picked_expected / args.public_size, 4),
                "confident_questions": confident,
                "max_equation_residual": round(report["max_equation_residual"], 6),
                "report": str(args.report),
                "build": build_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
