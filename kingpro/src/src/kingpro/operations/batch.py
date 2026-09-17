"""Crash-safe, ordered batch runner for the KINGPRO product pipeline.

The scoring artifact remains immutable.  This module wraps an answer function
with operational guarantees useful for private evaluation and Demo Day:

* validate the complete input before starting;
* execute concurrently while preserving input order;
* atomically checkpoint after every completed question;
* emit a self-contained trace for every question;
* record non-secret run and artifact fingerprints;
* resume a compatible interrupted run;
* use an exact-question fallback only when explicitly configured.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


AnswerFunction = Callable[[str], dict[str, Any]]
ProgressFunction = Callable[[dict[str, Any]], None]

_SENSITIVE_KEY = re.compile(r"(?:api.?key|token|secret|password|credential)", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "present": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "present": True,
        "size": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def _redact(value: Any, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(child): _redact(item, str(child)) for child, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def atomic_write_text(path: Path, text: str) -> None:
    """Write and fsync a temporary sibling before an atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
    temporary = path.with_name(path.name + suffix)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def append_report(path: Path, run_id: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"[{utc_now()}] [{run_id}] {message}\n")
        handle.flush()


def _normalize_question(value: Any) -> str:
    return " ".join(str(value or "").split())


def validate_questions(records: Sequence[Mapping[str, Any]], expected_count: int | None = None) -> list[dict[str, Any]]:
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"question count mismatch: expected={expected_count} actual={len(records)}")
    if not records:
        raise ValueError("question input is empty")

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"question {index} is not an object")
        if "id" not in record:
            raise ValueError(f"question {index} is missing id")
        identifier = record["id"]
        identifier_key = json.dumps(identifier, ensure_ascii=False, sort_keys=True)
        if identifier_key in seen:
            raise ValueError(f"duplicate question id: {identifier!r}")
        seen.add(identifier_key)
        question = _normalize_question(record.get("question"))
        if not question:
            raise ValueError(f"question {index} has empty question text")
        validated.append({"id": identifier, "question": question})
    return validated


def read_questions(path: Path, expected_count: int | None = None) -> list[dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix.casefold() == ".jsonl":
        records = [
            json.loads(line)
            for line in resolved.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
        if isinstance(payload, Mapping) and isinstance(payload.get("items"), list):
            records = payload["items"]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError("question JSON must be an array or an object with an items array")
    return validate_questions(records, expected_count=expected_count)


def load_fallback_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("fallback submission must be a JSON array")
    records: dict[str, dict[str, Any]] = {}
    for record in payload:
        if not isinstance(record, Mapping) or "id" not in record:
            continue
        key = json.dumps(record["id"], ensure_ascii=False, sort_keys=True)
        records[key] = dict(record)
    return records


@dataclass(frozen=True)
class BatchRunConfig:
    output_root: Path
    max_workers: int = 4
    expected_count: int | None = None
    fallback_on_refusal: bool = False
    run_id: str | None = None
    resume_run_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")


@dataclass(frozen=True)
class BatchRunPaths:
    run_id: str
    run_dir: Path
    manifest: Path
    running: Path
    final_success: Path
    final_warning: Path
    final_error: Path
    report: Path
    traces: Path


def _run_paths(config: BatchRunConfig) -> BatchRunPaths:
    if config.resume_run_dir is not None:
        run_dir = config.resume_run_dir.resolve()
        run_id = run_dir.name
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    else:
        run_id = config.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = (config.output_root / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    return BatchRunPaths(
        run_id=run_id,
        run_dir=run_dir,
        manifest=run_dir / "manifest.json",
        running=run_dir / "running.json",
        final_success=run_dir / "results_success.json",
        final_warning=run_dir / "results_warning.json",
        final_error=run_dir / "results_error.json",
        report=run_dir / "report.log",
        traces=run_dir / "traces",
    )


def _safe_trace_name(index: int, identifier: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(identifier)).strip("_") or "unknown"
    return f"q{index + 1:04d}_id_{token[:80]}.json"


def _fallback_for(
    fallback_records: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> dict[str, Any] | None:
    key = json.dumps(question["id"], ensure_ascii=False, sort_keys=True)
    candidate = fallback_records.get(key)
    if not candidate:
        return None
    if _normalize_question(candidate.get("question")) != question["question"]:
        return None
    return dict(candidate)


class BatchRunner:
    """Run an answer function with deterministic ordering and durable state."""

    def __init__(
        self,
        answer_fn: AnswerFunction,
        *,
        root: Path,
        input_path: Path,
        config: BatchRunConfig,
        fallback_records: Mapping[str, Mapping[str, Any]] | None = None,
        manifest_extra: Mapping[str, Any] | None = None,
        fingerprint_paths: Iterable[Path] = (),
        progress_fn: ProgressFunction | None = None,
    ) -> None:
        self.answer_fn = answer_fn
        self.root = root.resolve()
        self.input_path = input_path.resolve()
        self.config = config
        self.fallback_records = fallback_records or {}
        self.manifest_extra = _redact(dict(manifest_extra or {}))
        self.fingerprint_paths = [path.resolve() for path in fingerprint_paths]
        self.progress_fn = progress_fn

    def _initial_manifest(self, paths: BatchRunPaths, questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": paths.run_id,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "root": str(self.root),
            "input": {
                **file_fingerprint(self.input_path),
                "question_count": len(questions),
            },
            "runner": {
                "max_workers": self.config.max_workers,
                "expected_count": self.config.expected_count,
                "fallback_on_refusal": self.config.fallback_on_refusal,
                "ordered_output": True,
                "atomic_checkpoint": True,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
            "artifacts": [file_fingerprint(path) for path in self.fingerprint_paths],
            "extra": self.manifest_extra,
            "summary": {"completed": 0, "answered": 0, "refused": 0, "fallback": 0, "error": 0},
        }

    @staticmethod
    def _summary(results: Sequence[dict[str, Any] | None]) -> dict[str, int]:
        complete = [record for record in results if record is not None]
        summary = {"completed": len(complete), "answered": 0, "refused": 0, "fallback": 0, "error": 0}
        for record in complete:
            status = str(record.get("status", "error"))
            if status in summary:
                summary[status] += 1
            else:
                summary["error"] += 1
        return summary

    @staticmethod
    def _checkpoint_payload(paths: BatchRunPaths, results: Sequence[dict[str, Any] | None]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": paths.run_id,
            "status": "running",
            "updated_at": utc_now(),
            "summary": BatchRunner._summary(results),
            "results": list(results),
        }

    def _load_resume(
        self,
        paths: BatchRunPaths,
        questions: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any] | None]]:
        if not paths.manifest.is_file() or not paths.running.is_file():
            raise ValueError("resume directory must contain manifest.json and running.json")
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        checkpoint = json.loads(paths.running.read_text(encoding="utf-8"))
        current_hash = sha256_file(self.input_path)
        if manifest.get("input", {}).get("sha256") != current_hash:
            raise ValueError("resume input hash does not match the original run")
        results = checkpoint.get("results")
        if not isinstance(results, list) or len(results) != len(questions):
            raise ValueError("resume checkpoint result slots do not match question count")
        for index, record in enumerate(results):
            if record is None:
                continue
            if record.get("id") != questions[index]["id"] or record.get("question") != questions[index]["question"]:
                raise ValueError(f"resume checkpoint mismatch at index {index}")
        manifest["status"] = "running"
        manifest["resumed_at"] = utc_now()
        return manifest, results

    def _execute_one(self, index: int, question: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        started = time.perf_counter()
        started_at = utc_now()
        error: str | None = None
        fallback_used = False
        try:
            response = self.answer_fn(question["question"])
            if not isinstance(response, dict):
                raise TypeError("answer function must return a dict")
            response_status = str(response.get("status", "error"))
            should_fallback = response_status == "error" or (
                response_status == "refused" and self.config.fallback_on_refusal
            )
            if should_fallback:
                fallback = _fallback_for(self.fallback_records, question)
                if fallback is not None:
                    response = fallback
                    fallback_used = True
                    status = "fallback"
                else:
                    status = response_status
            else:
                status = response_status if response_status in {"answered", "refused"} else "answered"
        except Exception as exc:  # per-item isolation is intentional
            error = f"{type(exc).__name__}: {exc}"
            fallback = _fallback_for(self.fallback_records, question)
            if fallback is not None:
                response = fallback
                fallback_used = True
                status = "fallback"
            else:
                response = None
                status = "error"

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        record = {
            "index": index,
            "id": question["id"],
            "question": question["question"],
            "status": status,
            "fallback_used": fallback_used,
            "error": error,
            "started_at": started_at,
            "finished_at": utc_now(),
            "elapsed_ms": elapsed_ms,
            "trace_id": response.get("trace_id") if isinstance(response, dict) else None,
            "response": response,
        }
        return index, record

    def run(self, questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        validated = validate_questions(questions, expected_count=self.config.expected_count)
        paths = _run_paths(self.config)
        paths.traces.mkdir(parents=True, exist_ok=True)

        if self.config.resume_run_dir is not None:
            manifest, results = self._load_resume(paths, validated)
            append_report(paths.report, paths.run_id, f"RESUME completed={self._summary(results)['completed']}/{len(results)}")
        else:
            manifest = self._initial_manifest(paths, validated)
            results = [None] * len(validated)
            atomic_write_json(paths.manifest, manifest)
            atomic_write_json(paths.running, self._checkpoint_payload(paths, results))
            append_report(paths.report, paths.run_id, f"START total={len(results)} workers={self.config.max_workers}")

        pending = [index for index, record in enumerate(results) if record is None]
        executor = ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="kingpro-batch")
        fatal_error: str | None = None
        try:
            futures: dict[Future[tuple[int, dict[str, Any]]], int] = {
                executor.submit(self._execute_one, index, validated[index]): index
                for index in pending
            }
            for future in as_completed(futures):
                index, record = future.result()
                results[index] = record
                trace_path = paths.traces / _safe_trace_name(index, record["id"])
                atomic_write_json(trace_path, record)
                atomic_write_json(paths.running, self._checkpoint_payload(paths, results))
                summary = self._summary(results)
                append_report(
                    paths.report,
                    paths.run_id,
                    f"ITEM_DONE index={index + 1}/{len(results)} id={record['id']} status={record['status']} elapsed_ms={record['elapsed_ms']}",
                )
                if self.progress_fn is not None:
                    self.progress_fn({**summary, "total": len(results), "last": record})
        except KeyboardInterrupt:
            fatal_error = "KeyboardInterrupt"
            append_report(paths.report, paths.run_id, f"CANCELLED completed={self._summary(results)['completed']}/{len(results)}")
            raise
        except Exception as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"
            append_report(paths.report, paths.run_id, f"FATAL_ERROR {fatal_error}")
        finally:
            executor.shutdown(wait=fatal_error is None, cancel_futures=fatal_error is not None)

        summary = self._summary(results)
        manifest["summary"] = summary
        manifest["finished_at"] = utc_now()
        if fatal_error is not None or summary["completed"] != len(results):
            manifest["status"] = "error"
            manifest["fatal_error"] = fatal_error
            final_path = paths.final_error
        elif summary["fallback"] or summary["error"]:
            manifest["status"] = "warning"
            final_path = paths.final_warning
        else:
            manifest["status"] = "success"
            final_path = paths.final_success

        final_payload = {
            "schema_version": 1,
            "run_id": paths.run_id,
            "status": manifest["status"],
            "finished_at": manifest["finished_at"],
            "summary": summary,
            "results": list(results),
        }
        atomic_write_json(final_path, final_payload)
        atomic_write_json(paths.manifest, manifest)
        atomic_write_json(paths.running, self._checkpoint_payload(paths, results))
        append_report(paths.report, paths.run_id, f"FINISH status={manifest['status']} completed={summary['completed']}/{len(results)} output={final_path.name}")
        return {
            "run_id": paths.run_id,
            "status": manifest["status"],
            "run_dir": str(paths.run_dir),
            "output": str(final_path),
            "manifest": str(paths.manifest),
            "report": str(paths.report),
            "summary": summary,
        }
