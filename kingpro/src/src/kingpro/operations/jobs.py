"""Thread-safe durable batch jobs built on :mod:`kingpro.operations.batch`."""

from __future__ import annotations

import threading
import uuid
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import os

from kingpro.operations.batch import (
    BatchRunConfig,
    BatchRunner,
    atomic_write_json,
    load_fallback_records,
    utc_now,
    validate_questions,
)


TERMINAL_JOB_STATES = {"success", "warning", "error"}


def _tenant_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80] or "default"


@dataclass
class BatchJob:
    job_id: str
    questions: list[dict[str, Any]]
    workers: int
    input_path: Path
    tenant_id: str = "default"
    owner_id: str = "local-operator"
    created_at: str = field(default_factory=utc_now)
    status: str = "queued"
    summary: dict[str, int] = field(
        default_factory=lambda: {
            "completed": 0,
            "answered": 0,
            "refused": 0,
            "fallback": 0,
            "error": 0,
        }
    )
    run_dir: str | None = None
    output_path: str | None = None
    manifest_path: str | None = None
    report_path: str | None = None
    submission_dir: str | None = None
    submission_archive: str | None = None
    submission_manifest: str | None = None
    error: str | None = None
    finished_at: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def publish(self, event: str, data: Mapping[str, Any]) -> None:
        with self.condition:
            self.events.append(
                {
                    "seq": len(self.events),
                    "event": event,
                    "created_at": utc_now(),
                    "data": dict(data),
                }
            )
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "total": len(self.questions),
                "workers": self.workers,
                "tenant_id": self.tenant_id,
                "owner_id": self.owner_id,
                "summary": dict(self.summary),
                "run_dir": self.run_dir,
                "output_path": self.output_path,
                "manifest_path": self.manifest_path,
                "report_path": self.report_path,
                "download_ready": bool(self.output_path and Path(self.output_path).is_file()),
                "submission_ready": bool(
                    self.submission_archive and Path(self.submission_archive).is_file()
                ),
                "submission_archive": self.submission_archive,
                "submission_manifest": self.submission_manifest,
                "error": self.error,
            }


class BatchJobManager:
    """Create and observe durable background jobs in one product process."""

    def __init__(
        self,
        answer_fn: Callable[[str], dict[str, Any]],
        *,
        root: Path,
        output_root: Path,
        fallback_submission: Path | None = None,
        health_fn: Callable[[], dict[str, Any]] | None = None,
        fingerprint_paths: Iterable[Path] = (),
    ) -> None:
        self.answer_fn = answer_fn
        self.root = root.resolve()
        self.output_root = output_root.resolve()
        self.input_root = self.output_root / "inputs"
        self.run_root = self.output_root / "runs"
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.fallback_submission = fallback_submission.resolve() if fallback_submission else None
        self.fallback_records = load_fallback_records(self.fallback_submission)
        self.health_fn = health_fn
        self.fingerprint_paths = [path.resolve() for path in fingerprint_paths]
        self._jobs: dict[str, BatchJob] = {}
        self._lock = threading.RLock()

    def create(
        self,
        questions: Sequence[Mapping[str, Any]],
        *,
        workers: int = 4,
        fallback_on_refusal: bool = False,
        tenant_id: str = "default",
        owner_id: str = "local-operator",
    ) -> dict[str, Any]:
        if workers < 1 or workers > 16:
            raise ValueError("workers must be between 1 and 16")
        validated = validate_questions(questions)
        max_items = int(os.getenv("KINGPRO_MAX_BATCH_ITEMS", "2000"))
        if len(validated) > max_items:
            raise ValueError(f"batch contains {len(validated)} items; limit is {max_items}")
        max_active = int(os.getenv("KINGPRO_MAX_ACTIVE_JOBS_PER_TENANT", "2"))
        with self._lock:
            active = sum(
                job.tenant_id == str(tenant_id) and job.status not in TERMINAL_JOB_STATES
                for job in self._jobs.values()
            )
        if active >= max_active:
            raise ValueError("tenant active batch-job limit reached")
        job_id = "batch_{0}_{1}".format(utc_now()[0:19].replace(":", "").replace("-", ""), uuid.uuid4().hex[:8])
        tenant_slug = _tenant_slug(tenant_id)
        tenant_input_root = self.input_root / tenant_slug
        tenant_run_root = self.run_root / tenant_slug
        tenant_input_root.mkdir(parents=True, exist_ok=True)
        tenant_run_root.mkdir(parents=True, exist_ok=True)
        input_path = tenant_input_root / f"{job_id}.json"
        atomic_write_json(input_path, validated)
        job = BatchJob(
            job_id=job_id,
            questions=validated,
            workers=workers,
            input_path=input_path,
            tenant_id=str(tenant_id),
            owner_id=str(owner_id),
        )
        with self._lock:
            self._jobs[job_id] = job
        job.publish(
            "status",
            {
                "stage": "batch",
                "status": "queued",
                "message": f"Đã nhận {len(validated)} câu hỏi",
                "summary": dict(job.summary),
                "total": len(validated),
            },
        )
        thread = threading.Thread(
            target=self._run,
            args=(job, fallback_on_refusal),
            name=f"kingpro-{job_id}",
            daemon=True,
        )
        thread.start()
        return job.snapshot()

    def _run(self, job: BatchJob, fallback_on_refusal: bool) -> None:
        job.status = "running"
        job.publish(
            "status",
            {
                "stage": "batch",
                "status": "running",
                "message": "Batch job bắt đầu",
                "summary": dict(job.summary),
                "total": len(job.questions),
            },
        )

        def progress(payload: dict[str, Any]) -> None:
            job.summary = {
                key: int(payload.get(key, 0))
                for key in ("completed", "answered", "refused", "fallback", "error")
            }
            last = payload.get("last") or {}
            job.publish(
                "item",
                {
                    "stage": "batch_item",
                    "status": last.get("status"),
                    "message": f"Hoàn tất {job.summary['completed']}/{len(job.questions)}",
                    "summary": dict(job.summary),
                    "total": len(job.questions),
                    "last": {
                        "index": last.get("index"),
                        "id": last.get("id"),
                        "status": last.get("status"),
                        "elapsed_ms": last.get("elapsed_ms"),
                        "fallback_used": last.get("fallback_used"),
                        "error": last.get("error"),
                    },
                },
            )

        try:
            health = self.health_fn() if self.health_fn is not None else {}
            runner = BatchRunner(
                self.answer_fn,
                root=self.root,
                input_path=job.input_path,
                config=BatchRunConfig(
                    output_root=self.run_root / _tenant_slug(job.tenant_id),
                    max_workers=job.workers,
                    expected_count=len(job.questions),
                    fallback_on_refusal=fallback_on_refusal,
                    run_id=job.job_id,
                ),
                fallback_records=self.fallback_records,
                manifest_extra={
                    "service_health": health,
                    "job_id": job.job_id,
                    "tenant_id": job.tenant_id,
                    "owner_id": job.owner_id,
                    "fallback_submission": str(self.fallback_submission) if self.fallback_submission else None,
                },
                fingerprint_paths=[job.input_path, *self.fingerprint_paths],
                progress_fn=progress,
            )
            result = runner.run(job.questions)
            job.status = str(result["status"])
            job.summary = dict(result["summary"])
            job.run_dir = str(result["run_dir"])
            job.output_path = str(result["output"])
            job.manifest_path = str(result["manifest"])
            job.report_path = str(result["report"])
            job.finished_at = utc_now()
            job.publish("result", job.snapshot())
        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = utc_now()
            job.publish("error", job.snapshot())
        finally:
            job.publish("done", job.snapshot())

    def get(self, job_id: str, tenant_id: str | None = None) -> BatchJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and tenant_id is not None and job.tenant_id != tenant_id:
                return None
            return job

    def snapshot(self, job_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        job = self.get(job_id, tenant_id)
        return job.snapshot() if job else None

    def wait_events(
        self,
        job_id: str,
        cursor: int,
        *,
        timeout: float = 1.0,
        tenant_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        job = self.get(job_id, tenant_id)
        if job is None:
            raise KeyError(job_id)
        with job.condition:
            if len(job.events) <= cursor and job.status not in TERMINAL_JOB_STATES:
                job.condition.wait(timeout=timeout)
            events = [dict(event) for event in job.events[cursor:]]
            terminal = bool(
                job.status in TERMINAL_JOB_STATES
                and job.events
                and job.events[-1].get("event") == "done"
                and cursor + len(events) >= len(job.events)
            )
            return events, terminal

    def output_payload(self, job_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        job = self.get(job_id, tenant_id)
        if job is None or not job.output_path:
            return None
        path = Path(job.output_path)
        if not path.is_file():
            return None
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def export_submission(self, job_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        job = self.get(job_id, tenant_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in {"success", "warning"} or not job.output_path or not job.run_dir:
            raise ValueError("batch job is not ready for submission export")
        from kingpro.submission.export_batch import export_batch_results

        output_dir = Path(job.run_dir) / "submission_export"
        archive = Path(job.run_dir) / "kingpro_submission.zip"
        manifest_path = output_dir / "export_manifest.json"
        with self._lock:
            if archive.is_file() and manifest_path.is_file():
                import json

                report = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                report = export_batch_results(
                    Path(job.output_path),
                    root=self.root,
                    output_dir=output_dir,
                    archive_path=archive,
                    fallback_artifact=(
                        str(self.fallback_submission.parent.relative_to(self.root)).replace("\\", "/")
                        if self.fallback_submission
                        else "sub_v297_scope2"
                    ),
                    expected_count=len(job.questions),
                )
            job.submission_dir = str(output_dir)
            job.submission_archive = str(archive)
            job.submission_manifest = str(manifest_path)
        job.publish(
            "submission",
            {
                "status": "completed",
                "archive": str(archive),
                "sha256": report.get("archive", {}).get("sha256"),
            },
        )
        return report
