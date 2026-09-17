"""Strictly export durable product-batch results as a competition archive."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from kingpro.answering.sandbox import run_pandas_code
from kingpro.evaluation.metrics import coerce_number
from kingpro.operations.batch import atomic_write_json, file_fingerprint, utc_now
from kingpro.submission.archive import write_deterministic


_SAFE_FILE = re.compile(r"[^A-Za-z0-9_.-]+")
_VARIABLE = re.compile(r"^df[1-9]\d*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"evidence source is outside project root: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _source_path(root: Path, contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> Path:
    if evidence.get("source_path"):
        value = Path(str(evidence["source_path"]))
        return _inside(root, value if value.is_absolute() else root / value)
    source_artifact = Path(str(contract.get("source_artifact") or "."))
    csv_path = Path(str(evidence.get("csv_path") or ""))
    if not str(csv_path):
        raise ValueError("submission evidence is missing source_path/csv_path")
    base = source_artifact if source_artifact.is_absolute() else root / source_artifact
    return _inside(root, base / csv_path)


def _contract(record: Mapping[str, Any], fallback_artifact: str) -> dict[str, Any]:
    status = str(record.get("status") or "error")
    response = record.get("response")
    if status not in {"answered", "fallback"} or not isinstance(response, Mapping):
        raise ValueError(f"question {record.get('id')} cannot export status={status}")
    contract = response.get("submission_export")
    if isinstance(contract, Mapping):
        return dict(contract)
    required = {"answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"}
    if required <= set(response):
        return {
            "source_artifact": fallback_artifact,
            **{key: response[key] for key in required},
        }
    raise ValueError(f"question {record.get('id')} has no submission export contract")


def _safe_destination(source: Path, digest: str, occupied: dict[str, str]) -> str:
    clean = _SAFE_FILE.sub("_", source.name).strip("._") or "evidence.csv"
    if not clean.casefold().endswith(".csv"):
        clean += ".csv"
    key = clean.casefold()
    existing = occupied.get(key)
    if existing is None or existing == digest:
        occupied[key] = digest
        return clean
    stem = Path(clean).stem[:80]
    candidate = f"{stem}_{digest[:12]}.csv"
    key = candidate.casefold()
    if key in occupied and occupied[key] != digest:
        raise ValueError(f"unresolvable evidence filename collision: {source.name}")
    occupied[key] = digest
    return candidate


def _validate_and_stage(
    batch: Mapping[str, Any],
    *,
    root: Path,
    staging: Path,
    fallback_artifact: str,
    expected_count: int | None,
    execution_timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = batch.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("batch result must contain a non-empty results array")
    if expected_count is not None and len(results) != expected_count:
        raise ValueError(f"result count mismatch: expected={expected_count} actual={len(results)}")
    if any(record is None for record in results):
        raise ValueError("batch result contains incomplete slots")

    data_dir = staging / "data"
    data_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    copied_by_hash: dict[str, str] = {}
    occupied: dict[str, str] = {}
    execution_checks: list[dict[str, Any]] = []

    for position, raw_record in enumerate(results, start=1):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"result {position} is not an object")
        identifier = raw_record.get("id")
        identifier_key = json.dumps(identifier, ensure_ascii=False, sort_keys=True)
        if identifier_key in seen_ids:
            raise ValueError(f"duplicate result id: {identifier!r}")
        seen_ids.add(identifier_key)
        question = " ".join(str(raw_record.get("question") or "").split())
        if not question:
            raise ValueError(f"result {position} has empty question")
        contract = _contract(raw_record, fallback_artifact)
        answer = coerce_number(contract.get("answer"))
        if answer is None or not math.isfinite(float(answer)):
            raise ValueError(f"result {identifier} answer is not a finite scalar")
        pandas_query = str(contract.get("pandas_query") or "")
        if not pandas_query.strip():
            raise ValueError(f"result {identifier} has empty pandas_query")
        relevant_docs = [str(value) for value in contract.get("relevant_docs", []) if str(value)]
        relevant_tables = [str(value) for value in contract.get("relevant_tables", []) if str(value)]
        if not relevant_docs or not relevant_tables:
            raise ValueError(f"result {identifier} is missing relevant docs/tables")
        evidence_items = contract.get("evidence")
        if not isinstance(evidence_items, list) or not evidence_items:
            raise ValueError(f"result {identifier} has no evidence")

        exported_evidence: list[dict[str, str]] = []
        runtime_paths: dict[str, str] = {}
        variables: set[str] = set()
        for evidence in evidence_items:
            if not isinstance(evidence, Mapping):
                raise ValueError(f"result {identifier} has invalid evidence entry")
            variable = str(evidence.get("variable") or "")
            if not _VARIABLE.fullmatch(variable) or variable in variables:
                raise ValueError(f"result {identifier} has invalid/duplicate variable {variable!r}")
            variables.add(variable)
            source = _source_path(root, contract, evidence)
            digest = _sha256(source)
            destination_name = copied_by_hash.get(digest)
            if destination_name is None:
                destination_name = _safe_destination(source, digest, occupied)
                shutil.copyfile(source, data_dir / destination_name)
                copied_by_hash[digest] = destination_name
            relative = f"data/{destination_name}"
            exported_evidence.append({"variable": variable, "csv_path": relative})
            runtime_paths[variable] = str((data_dir / destination_name).resolve())

        replay = run_pandas_code(pandas_query, runtime_paths, timeout=execution_timeout)
        replay_number = coerce_number(replay.get("result")) if replay.get("ok") else None
        tolerance = 1e-6 + 1e-9 * max(abs(float(answer)), abs(float(replay_number or 0)), 1.0)
        if replay_number is None or not math.isfinite(float(replay_number)):
            raise ValueError(f"result {identifier} export replay failed: {replay.get('error')}")
        if abs(float(answer) - float(replay_number)) > tolerance:
            raise ValueError(
                f"result {identifier} export replay mismatch: answer={answer} replay={replay_number}"
            )
        execution_checks.append(
            {"id": identifier, "answer": float(answer), "replay": float(replay_number), "match": True}
        )
        rows.append(
            {
                "id": identifier,
                "question": question,
                "answer": float(answer),
                "relevant_docs": list(dict.fromkeys(relevant_docs)),
                "relevant_tables": list(dict.fromkeys(relevant_tables)),
                "evidence": exported_evidence,
                "pandas_query": pandas_query,
            }
        )

    if not any(data_dir.glob("*.csv")):
        raise ValueError("export produced no evidence CSV")
    return rows, {
        "question_count": len(rows),
        "evidence_file_count": len(list(data_dir.glob("*.csv"))),
        "execution_checks": execution_checks,
    }


def export_batch_results(
    batch_path: Path,
    *,
    root: Path,
    output_dir: Path,
    archive_path: Path,
    fallback_artifact: str = "sub_v297_scope2",
    expected_count: int | None = None,
    execution_timeout: float = 8.0,
) -> dict[str, Any]:
    root = root.resolve()
    batch_path = batch_path.resolve()
    output_dir = output_dir.resolve()
    archive_path = archive_path.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if archive_path.exists():
        raise FileExistsError(archive_path)
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("batch output must be a JSON object")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent)))
    temporary_archive = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.tmp")
    try:
        rows, validation = _validate_and_stage(
            payload,
            root=root,
            staging=staging,
            fallback_artifact=fallback_artifact,
            expected_count=expected_count,
            execution_timeout=execution_timeout,
        )
        atomic_write_json(staging / "submission.json", rows)
        with zipfile.ZipFile(
            temporary_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as bundle:
            write_deterministic(bundle, staging / "submission.json", "submission.json")
            for csv_path in sorted((staging / "data").glob("*.csv"), key=lambda path: path.name.casefold()):
                write_deterministic(bundle, csv_path, f"data/{csv_path.name}")
        with zipfile.ZipFile(temporary_archive, "r") as bundle:
            corrupt = bundle.testzip()
            if corrupt:
                raise ValueError(f"archive CRC failure at {corrupt}")
            names = bundle.namelist()
            if len(names) != len(set(names)) or "submission.json" not in names:
                raise ValueError("archive layout/duplicate validation failed")
        staging.replace(output_dir)
        os.replace(temporary_archive, archive_path)
        manifest = {
            "schema_version": 1,
            "kind": "batch_submission_export",
            "created_at": utc_now(),
            "source_batch": file_fingerprint(batch_path),
            "output_dir": str(output_dir),
            "archive": file_fingerprint(archive_path),
            "submission": file_fingerprint(output_dir / "submission.json"),
            "validation": validation,
            "status": "PASS",
        }
        atomic_write_json(output_dir / "export_manifest.json", manifest)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if temporary_archive.exists():
            temporary_archive.unlink()
        raise
