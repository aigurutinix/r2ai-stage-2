"""Build controlled ViFinQA leaderboard ablations from immutable ZIP files.

The public leaderboard is useful only when every experiment has a precise
change set.  This module treats a known-good submission ZIP as immutable and
copies candidate objects/evidence for an explicit list of question IDs.  It
also refuses evidence changes that would leak into questions outside the
declared experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional


SUBMISSION_FIELDS = {
    "id",
    "question",
    "answer",
    "relevant_docs",
    "relevant_tables",
    "evidence",
    "pandas_query",
}
EXPERIMENT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer a repository-relative path in portable manifests."""

    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        name == path.as_posix()
        and not path.is_absolute()
        and ".." not in path.parts
        and name != ""
        and not name.endswith("/")
    )


def _evidence_paths(item: dict[str, Any]) -> set[str]:
    return {str(evidence["csv_path"]) for evidence in item["evidence"]}


def _validate_item(item: Any, source: str) -> dict[str, Any]:

    if not isinstance(item, dict) or set(item) != SUBMISSION_FIELDS:
        raise ValueError(f"{source}: submission fields do not match the schema")
    question_id = item["id"]
    if not isinstance(question_id, int) or isinstance(question_id, bool):
        raise ValueError(f"{source}: id must be an integer")
    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValueError(f"{source}: question must be a non-empty string")
    if (
        not isinstance(item["answer"], (int, float))
        or isinstance(item["answer"], bool)
        or not math.isfinite(float(item["answer"]))
    ):
        raise ValueError(f"{source}: answer must be a finite number")
    for field in ("relevant_docs", "relevant_tables"):
        if not isinstance(item[field], list) or not all(
            isinstance(value, str) and value for value in item[field]
        ):
            raise ValueError(f"{source}: {field} must be a non-empty string list")
    if not item["relevant_docs"] or not item["relevant_tables"]:
        raise ValueError(f"{source}: retrieval provenance cannot be empty")
    if not isinstance(item["pandas_query"], str) or not item["pandas_query"].strip():
        raise ValueError(f"{source}: pandas_query must be a non-empty string")
    if not isinstance(item["evidence"], list) or not item["evidence"]:
        raise ValueError(f"{source}: evidence must be a non-empty list")
    variables: set[str] = set()
    for evidence in item["evidence"]:
        if not isinstance(evidence, dict) or set(evidence) != {"variable", "csv_path"}:
            raise ValueError(f"{source}: invalid evidence object")
        variable = evidence["variable"]
        path = evidence["csv_path"]
        if not isinstance(variable, str) or not variable.isidentifier():
            raise ValueError(f"{source}: invalid evidence variable {variable!r}")
        if variable in variables:
            raise ValueError(f"{source}: duplicate evidence variable {variable!r}")
        variables.add(variable)
        if (
            not isinstance(path, str)
            or not path.startswith("data/")
            or not path.endswith(".csv")
            or not _safe_member(path)
        ):
            raise ValueError(f"{source}: unsafe evidence path {path!r}")
    return item


@dataclass(frozen=True)
class SubmissionArchive:
    path: Path
    items: list[dict[str, Any]]
    by_id: dict[int, dict[str, Any]]
    members: dict[str, bytes]

    @classmethod
    def load(cls, path: Path, *, require_all_evidence: bool = True) -> "SubmissionArchive":
        if not path.is_file():
            raise FileNotFoundError(path)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"{path}: duplicate ZIP members")
            unsafe = [name for name in names if not _safe_member(name)]
            if unsafe:
                raise ValueError(f"{path}: unsafe ZIP members: {unsafe[:3]}")
            if "submission.json" not in names:
                raise ValueError(f"{path}: missing submission.json")
            members = {name: archive.read(name) for name in names}
        try:
            payload = json.loads(members["submission.json"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}: invalid submission.json: {error}") from error
        if not isinstance(payload, list):
            raise ValueError(f"{path}: submission.json must be a JSON array")
        items: list[dict[str, Any]] = []
        by_id: dict[int, dict[str, Any]] = {}
        for index, raw_item in enumerate(payload):
            item = _validate_item(raw_item, f"{path}: item {index}")
            question_id = item["id"]
            if question_id in by_id:
                raise ValueError(f"{path}: duplicate question id {question_id}")
            items.append(item)
            by_id[question_id] = item
        if require_all_evidence:
            missing = sorted(
                path_name
                for item in items
                for path_name in _evidence_paths(item)
                if path_name not in members
            )
            if missing:
                raise ValueError(f"{path}: missing evidence member {missing[0]!r}")
        return cls(path=path, items=items, by_id=by_id, members=members)


def parse_question_ids(values: Iterable[str]) -> set[int]:
    result: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip().lower().removeprefix("q")
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                start, end = int(left), int(right)
                if start > end:
                    raise ValueError(f"Invalid descending ID range: {token}")
                result.update(range(start, end + 1))
            else:
                result.add(int(token))
    if not result or min(result) < 1:
        raise ValueError("At least one positive question ID is required")
    return result


def _changed_fields(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    return sorted(field for field in SUBMISSION_FIELDS if baseline[field] != candidate[field])


def _usage_by_path(items: Iterable[dict[str, Any]]) -> dict[str, set[int]]:
    usage: dict[str, set[int]] = {}
    for item in items:
        for path in _evidence_paths(item):
            usage.setdefault(path, set()).add(item["id"])
    return usage


def _zip_bytes(path: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info, data


def _write_benchmark_locked_zip(
    output_zip: Path,
    submission: list[dict[str, Any]],
    evidence: dict[str, bytes],
) -> None:
    submission_bytes = (
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    with zipfile.ZipFile(output_zip, "w") as archive:
        info, data = _zip_bytes("submission.json", submission_bytes)
        archive.writestr(info, data)
        for path in sorted(evidence):
            info, data = _zip_bytes(path, evidence[path])
            archive.writestr(info, data)


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number}: event must be an object")
            events.append(event)
    return events


def append_ledger(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_ablation(
    *,
    baseline_zip: Path,
    candidate_zip: Path,
    question_ids: set[int],
    output_zip: Path,
    experiment_id: str,
    ledger_path: Optional[Path] = None,
    baseline_score: Optional[float] = None,
    notes: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Create an archive that differs from baseline only for authorized IDs."""

    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(f"Invalid experiment id: {experiment_id!r}")
    baseline_zip = baseline_zip.resolve()
    candidate_zip = candidate_zip.resolve()
    output_zip = output_zip.resolve()
    if output_zip in {baseline_zip, candidate_zip}:
        raise ValueError("Output ZIP must not overwrite an input ZIP")
    if output_zip.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_zip}")
    if ledger_path is not None:
        ledger_path = ledger_path.resolve()
        if any(
            event.get("event") == "build"
            and event.get("experiment_id") == experiment_id
            for event in _read_ledger(ledger_path)
        ):
            raise ValueError(f"Experiment already exists in ledger: {experiment_id}")

    baseline = SubmissionArchive.load(baseline_zip)
    candidate = SubmissionArchive.load(candidate_zip)
    missing_baseline = sorted(question_ids - baseline.by_id.keys())
    missing_candidate = sorted(question_ids - candidate.by_id.keys())
    if missing_baseline:
        raise ValueError(f"IDs missing from baseline: {missing_baseline[:10]}")
    if missing_candidate:
        raise ValueError(f"IDs missing from candidate: {missing_candidate[:10]}")

    changed_fields: dict[str, list[str]] = {}
    merged_by_id = dict(baseline.by_id)
    for question_id in sorted(question_ids):
        old = baseline.by_id[question_id]
        new = candidate.by_id[question_id]
        if new["question"] != old["question"]:
            raise ValueError(f"q{question_id}: candidate changes the question text")
        fields = _changed_fields(old, new)
        if fields:
            changed_fields[str(question_id)] = fields
        merged_by_id[question_id] = new
    merged = [merged_by_id[item["id"]] for item in baseline.items]
    usage = _usage_by_path(merged)

    authorized_candidate_paths = {
        path
        for question_id in question_ids
        for path in _evidence_paths(candidate.by_id[question_id])
    }
    changed_evidence: set[str] = set()
    for path in authorized_candidate_paths:
        if path not in candidate.members:
            if path not in baseline.members:
                raise ValueError(f"Candidate evidence is missing from both ZIPs: {path}")
            continue
        if baseline.members.get(path) != candidate.members[path]:
            changed_evidence.add(path)

    unauthorized_impacts: dict[str, list[int]] = {}
    for path in sorted(changed_evidence):
        outside = sorted(usage.get(path, set()) - question_ids)
        if outside:
            unauthorized_impacts[path] = outside
    if unauthorized_impacts:
        path, ids = next(iter(unauthorized_impacts.items()))
        raise ValueError(
            f"Evidence collision: {path} also affects unauthorized IDs {ids[:10]}"
        )

    evidence: dict[str, bytes] = {}
    required_paths = set(usage)
    for path in sorted(required_paths):
        if path in changed_evidence:
            evidence[path] = candidate.members[path]
        elif path in baseline.members:
            evidence[path] = baseline.members[path]
        elif path in candidate.members:
            evidence[path] = candidate.members[path]
        else:
            raise ValueError(f"Output evidence is missing from both ZIPs: {path}")

    evidence_impacted_ids = {
        question_id
        for path in changed_evidence
        for question_id in usage.get(path, set())
    }
    effective_ids = set(map(int, changed_fields)) | evidence_impacted_ids
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    if not effective_ids:
        shutil.copyfile(baseline_zip, output_zip)
    else:
        _write_benchmark_locked_zip(output_zip, merged, evidence)

    built = SubmissionArchive.load(output_zip)
    if set(built.by_id) != set(baseline.by_id):
        raise AssertionError("Output ID coverage differs from baseline")
    for question_id in baseline.by_id:
        expected = candidate.by_id[question_id] if question_id in question_ids else baseline.by_id[question_id]
        if built.by_id[question_id] != expected:
            raise AssertionError(f"q{question_id}: output item differs from its expected source")
    if set(built.members) != {"submission.json", *required_paths}:
        raise AssertionError("Output ZIP members differ from referenced evidence")
    for path, expected_bytes in evidence.items():
        if built.members[path] != expected_bytes:
            raise AssertionError(f"Output evidence bytes differ for {path}")

    manifest = {
        "schema_version": 1,
        "event": "build",
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "baseline_zip": display_path(baseline_zip),
        "baseline_sha256": sha256_file(baseline_zip),
        "baseline_execution_accuracy": baseline_score,
        "candidate_zip": display_path(candidate_zip),
        "candidate_sha256": sha256_file(candidate_zip),
        "output_zip": display_path(output_zip),
        "output_sha256": sha256_file(output_zip),
        "requested_ids": sorted(question_ids),
        "effective_changed_ids": sorted(effective_ids),
        "changed_fields": changed_fields,
        "changed_evidence": sorted(changed_evidence),
        "unchanged_item_count": len(baseline.items) - len(changed_fields),
        "item_count": len(built.items),
        "evidence_file_count": len(required_paths),
        "structural_validation": "PASS",
        "execution_validation": "NOT_RUN",
        "notes": notes,
    }
    manifest_path = output_zip.with_suffix(".manifest.json")
    if manifest_path.exists() and not force:
        output_zip.unlink()
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if ledger_path is not None:
        append_ledger(ledger_path, manifest)
    return manifest


def _parse_metrics(value: str) -> dict[str, float]:
    path = Path(value)
    raw = path.read_text(encoding="utf-8") if path.is_file() else value
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Metrics must be a JSON object")
    metrics: dict[str, float] = {}
    for key, metric in payload.items():
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            raise ValueError(f"Invalid metric {key}: {metric!r}")
        metrics[str(key)] = float(metric)
    if "EXECUTION_ACCURACY" not in metrics:
        raise ValueError("Metrics must include EXECUTION_ACCURACY")
    return metrics


def record_score(
    ledger_path: Path,
    experiment_id: str,
    metrics: dict[str, float],
    notes: str = "",
) -> dict[str, Any]:
    events = _read_ledger(ledger_path)
    builds = [
        event
        for event in events
        if event.get("event") == "build" and event.get("experiment_id") == experiment_id
    ]
    if len(builds) != 1:
        raise ValueError(f"Expected exactly one build event for {experiment_id!r}")
    if any(
        event.get("event") == "score" and event.get("experiment_id") == experiment_id
        for event in events
    ):
        raise ValueError(f"Score already recorded for {experiment_id!r}")
    baseline_score = builds[0].get("baseline_execution_accuracy")
    execution_accuracy = metrics["EXECUTION_ACCURACY"]
    event = {
        "schema_version": 1,
        "event": "score",
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "metrics": metrics,
        "execution_accuracy_delta": (
            execution_accuracy - float(baseline_score)
            if baseline_score is not None
            else None
        ),
        "notes": notes,
    }
    append_ledger(ledger_path, event)
    return event


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build one controlled A/B ZIP")
    build.add_argument("--baseline-zip", type=Path, required=True)
    build.add_argument("--candidate-zip", type=Path, required=True)
    build.add_argument("--ids", nargs="+", required=True, help="IDs/ranges, e.g. 387 434 or 362-380")
    build.add_argument("--output-zip", type=Path, required=True)
    build.add_argument("--experiment-id", required=True)
    build.add_argument("--ledger", type=Path, default=Path("analysis/experiment_ledger.jsonl"))
    build.add_argument("--baseline-score", type=float, default=0.5968)
    build.add_argument("--notes", default="")
    build.add_argument("--force", action="store_true")

    score = subparsers.add_parser("score", help="Append leaderboard metrics")
    score.add_argument("--ledger", type=Path, default=Path("analysis/experiment_ledger.jsonl"))
    score.add_argument("--experiment-id", required=True)
    score.add_argument("--metrics", required=True, help="Inline JSON or path to JSON")
    score.add_argument("--notes", default="")

    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_ablation(
            baseline_zip=args.baseline_zip,
            candidate_zip=args.candidate_zip,
            question_ids=parse_question_ids(args.ids),
            output_zip=args.output_zip,
            experiment_id=args.experiment_id,
            ledger_path=args.ledger,
            baseline_score=args.baseline_score,
            notes=args.notes,
            force=args.force,
        )
    else:
        result = record_score(
            ledger_path=args.ledger,
            experiment_id=args.experiment_id,
            metrics=_parse_metrics(args.metrics),
            notes=args.notes,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
