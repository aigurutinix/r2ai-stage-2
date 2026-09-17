"""Dependency-light HTTP API for the KINGPRO live product.

Endpoints:
  GET  /health
  GET  /catalog
  POST /ask       {"question": "..."}
  POST /ask/stream {"question": "..."}  # SSE status/result/done
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from _env import load_local_env  # noqa: E402

load_local_env(ROOT)

from kingpro.product import ProductService  # noqa: E402
from kingpro.operations import BatchJobManager  # noqa: E402
from kingpro.governance.approval import approval_contract  # noqa: E402
from kingpro.governance.audit import get_audit_ledger  # noqa: E402
from kingpro.governance.auth import AuthError, AuthPolicy, Principal  # noqa: E402
from kingpro.governance.upload import UploadQuarantine, UploadRejected  # noqa: E402
from kingpro.governance.rate_limit import RateLimiter  # noqa: E402
from kingpro.governance.metrics import METRICS  # noqa: E402
from kingpro.governance.feedback import FeedbackStore  # noqa: E402

SERVICE = ProductService(root=ROOT)
MAX_BODY_BYTES = 64 * 1024
BATCH_MAX_BODY_BYTES = 2 * 1024 * 1024
UPLOAD_MAX_BODY_BYTES = int(os.getenv("KINGPRO_UPLOAD_MAX_BYTES", str(25 * 1024 * 1024)))
ALLOWED_ORIGIN = os.getenv("KINGPRO_ALLOWED_ORIGIN", "http://127.0.0.1:3000")
JOB_MANAGER = BatchJobManager(
    SERVICE.ask,
    root=ROOT,
    output_root=ROOT / "outputs" / "product_batch_jobs",
    fallback_submission=ROOT / "sub_v297_scope2" / "submission.json",
    health_fn=SERVICE.health,
    fingerprint_paths=(
        ROOT / "sub_v297_scope2" / "submission.json",
        ROOT / "build" / "catalog.jsonl",
        ROOT / "build" / "bm25" / "params.index.json",
        ROOT / "build" / "bm25" / "vocab.index.json",
        ROOT / "src" / "kingpro" / "product" / "service.py",
    ),
)
AUTH_POLICY = AuthPolicy()
UPLOAD_MANAGER = UploadQuarantine(ROOT / "outputs" / "uploads")
RATE_LIMITER = RateLimiter()
FEEDBACK_STORE = FeedbackStore(ROOT / "outputs" / "feedback")


class Handler(BaseHTTPRequestHandler):
    server_version = "KINGPRO/1.0"

    def setup(self) -> None:
        super().setup()
        self._request_started = time.perf_counter()
        self._response_status = 0

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            METRICS.observe(
                getattr(self, "command", "UNKNOWN"),
                getattr(self, "path", "unknown"),
                self._response_status or 500,
                (time.perf_counter() - self._request_started) * 1000,
            )

    def _headers(self, status: int = 200) -> None:
        self._response_status = int(status)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Kingpro-Approval",
        )
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        self.end_headers()

    def _json(self, payload: dict, status: int = 200) -> None:
        self._headers(status)
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _binary(self, path: Path, filename: str) -> None:
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _sse_headers(self) -> None:
        self._response_status = int(HTTPStatus.OK)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()

    def _require(self, role: str) -> Principal | None:
        try:
            principal = AUTH_POLICY.authenticate(self.headers, role)
        except AuthError as exc:
            get_audit_ledger().append(
                "api.authorization_denied",
                {"path": self.path, "method": self.command, "role": role, "code": exc.code},
            )
            self._json(
                {"status": "error", "error": exc.code, "message": str(exc)},
                exc.status,
            )
            return None
        policy = {
            "viewer": (120, 60.0),
            "analyst": (40, 60.0),
            "approver": (20, 60.0),
            "admin": (120, 60.0),
        }[role]
        allowed, retry = RATE_LIMITER.allow(
            f"{principal.tenant_id}:{principal.subject}",
            role,
            capacity=policy[0],
            per_seconds=policy[1],
        )
        if not allowed:
            get_audit_ledger().append(
                "api.rate_limited",
                {
                    "path": self.path.split("?", 1)[0],
                    "subject": principal.subject,
                    "tenant_id": principal.tenant_id,
                    "role": role,
                    "retry_after_seconds": round(retry, 3),
                },
            )
            self._json(
                {"status": "error", "error": "rate_limited", "retry_after_seconds": retry},
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            return None
        get_audit_ledger().append(
            "api.authorized",
            {
                "path": self.path.split("?", 1)[0],
                "method": self.command,
                "required_role": role,
                "subject": principal.subject,
                "tenant_id": principal.tenant_id,
                "auth_mode": principal.auth_mode,
            },
        )
        return principal

    def _stream_question(self, question: str, principal: Principal) -> None:
        events: queue.Queue[tuple[str, dict]] = queue.Queue()

        def on_progress(payload: dict) -> None:
            events.put(("status", payload))

        def run() -> None:
            try:
                result = SERVICE.ask(question, progress_fn=on_progress)
                result["principal"] = principal.public()
                events.put(("result", result))
            except Exception as exc:
                events.put(
                    (
                        "error",
                        {
                            "stage": "request",
                            "title": "Request failed",
                            "status": "error",
                            "elapsed_ms": 0,
                            "detail": f"{type(exc).__name__}: {str(exc)[:200]}",
                            "metadata": {},
                        },
                    )
                )
            finally:
                events.put(("done", {"status": "completed"}))

        self._sse_headers()
        threading.Thread(target=run, name="kingpro-ask-stream", daemon=True).start()
        current_stage = "request"
        current_title = "Đang xử lý câu hỏi"
        heartbeat_started = time.perf_counter()
        try:
            while True:
                try:
                    event, payload = events.get(timeout=1.0)
                except queue.Empty:
                    event = "status"
                    payload = {
                        "stage": current_stage,
                        "title": current_title,
                        "status": "running",
                        "elapsed_ms": int((time.perf_counter() - heartbeat_started) * 1000),
                        "detail": "Backend vẫn đang xử lý; kết nối SSE còn hoạt động.",
                        "metadata": {"heartbeat": True},
                    }
                if event == "status":
                    current_stage = str(payload.get("stage") or current_stage)
                    current_title = str(payload.get("title") or current_title)
                    if payload.get("status") == "started":
                        heartbeat_started = time.perf_counter()
                message = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
                if event == "done":
                    break
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            self.close_connection = True

    def _stream_batch_events(self, job_id: str, cursor: int, principal: Principal) -> None:
        self._sse_headers()
        try:
            while True:
                events, terminal = JOB_MANAGER.wait_events(
                    job_id, cursor, timeout=1.0, tenant_id=principal.tenant_id
                )
                if not events:
                    snapshot = JOB_MANAGER.snapshot(job_id) or {"job_id": job_id, "status": "missing"}
                    events = [
                        {
                            "seq": None,
                            "event": "status",
                            "data": {
                                "stage": "batch",
                                "status": snapshot.get("status"),
                                "message": "Batch job vẫn đang chạy; kết nối SSE còn hoạt động.",
                                "summary": snapshot.get("summary", {}),
                                "total": snapshot.get("total", 0),
                                "heartbeat": True,
                            },
                        }
                    ]
                for record in events:
                    event = str(record.get("event") or "status")
                    data = dict(record.get("data") or {})
                    data["seq"] = record.get("seq")
                    event_id = record.get("seq")
                    id_line = f"id: {event_id}\n" if isinstance(event_id, int) else ""
                    message = f"{id_line}event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    self.wfile.write(message.encode("utf-8"))
                    self.wfile.flush()
                    if isinstance(record.get("seq"), int):
                        cursor = int(record["seq"]) + 1
                if terminal:
                    break
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            self.close_connection = True

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._headers(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path == "/health":
            self._json({**SERVICE.health(), "auth": AUTH_POLICY.health()})
            return
        if path == "/metrics":
            if self._require("admin") is None:
                return
            self._json(METRICS.snapshot())
            return
        if path == "/catalog":
            if self._require("viewer") is None:
                return
            self._json(SERVICE.catalog_summary())
            return
        upload_match = re.fullmatch(r"/uploads/(upl_[A-Fa-f0-9]{32})", path)
        if upload_match:
            principal = self._require("analyst")
            if principal is None:
                return
            payload = UPLOAD_MANAGER.status(upload_match.group(1), principal.tenant_id)
            if payload is None:
                self._json({"status": "not_found"}, HTTPStatus.NOT_FOUND)
                return
            self._json(payload)
            return
        match = re.fullmatch(
            r"/batch/jobs/([A-Za-z0-9_.-]+)(?:/(events|download|submission/download))?",
            path,
        )
        if match:
            job_id, action = match.groups()
            required_role = "approver" if action == "submission/download" else "analyst"
            principal = self._require(required_role)
            if principal is None:
                return
            snapshot = JOB_MANAGER.snapshot(job_id, principal.tenant_id)
            if snapshot is None:
                self._json({"status": "not_found", "job_id": job_id}, HTTPStatus.NOT_FOUND)
                return
            if action == "events":
                query = parse_qs(parsed.query)
                raw_cursor = query.get("cursor", [self.headers.get("Last-Event-ID", "0")])[0]
                try:
                    cursor = max(0, int(raw_cursor))
                except (TypeError, ValueError):
                    cursor = 0
                self._stream_batch_events(job_id, cursor, principal)
                return
            if action == "download":
                payload = JOB_MANAGER.output_payload(job_id, principal.tenant_id)
                if payload is None:
                    self._json({"status": "not_ready", "job_id": job_id}, HTTPStatus.CONFLICT)
                    return
                self._json(payload)
                return
            if action == "submission/download":
                archive = snapshot.get("submission_archive")
                if not snapshot.get("submission_ready") or not archive:
                    self._json({"status": "not_ready", "job_id": job_id}, HTTPStatus.CONFLICT)
                    return
                self._binary(Path(str(archive)), "kingpro_submission.zip")
                return
            self._json(snapshot)
            return
        self._json({"status": "not_found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        feedback_adjudication = re.fullmatch(r"/feedback/(fb_[A-Fa-f0-9]{32})/adjudicate", path)
        if path == "/feedback" or feedback_adjudication:
            principal = self._require("approver" if feedback_adjudication else "viewer")
            if principal is None:
                return
            if feedback_adjudication and self.headers.get(
                "X-Kingpro-Approval", ""
            ).strip().lower() != "user-confirmed":
                self._json(
                    {"status": "approval_required", "approval": approval_contract("publish_report")},
                    HTTPStatus.FORBIDDEN,
                )
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = -1
            if size < 1 or size > MAX_BODY_BYTES:
                self._json({"status": "error", "error": "request_too_large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            try:
                body = json.loads(self.rfile.read(size).decode("utf-8"))
                if feedback_adjudication:
                    result = FEEDBACK_STORE.adjudicate(
                        tenant_id=principal.tenant_id,
                        actor=principal.subject,
                        feedback_id=feedback_adjudication.group(1),
                        decision=str(body.get("decision") or ""),
                        evidence_refs=list(body.get("evidence_refs") or []),
                    )
                else:
                    result = FEEDBACK_STORE.submit(
                        tenant_id=principal.tenant_id,
                        actor=principal.subject,
                        trace_id=str(body.get("trace_id") or ""),
                        verdict=str(body.get("verdict") or ""),
                        reason_code=str(body.get("reason_code") or ""),
                        comment=str(body.get("comment") or ""),
                    )
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                self._json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"status": "ok", **result})
            return
        approve_upload = re.fullmatch(r"/uploads/(upl_[A-Fa-f0-9]{32})/approve", path)
        if approve_upload:
            principal = self._require("approver")
            if principal is None:
                return
            if self.headers.get("X-Kingpro-Approval", "").strip().lower() != "user-confirmed":
                self._json(
                    {"status": "approval_required", "approval": approval_contract("publish_report")},
                    HTTPStatus.FORBIDDEN,
                )
                return
            try:
                metadata = UPLOAD_MANAGER.approve(
                    approve_upload.group(1), principal.tenant_id, principal.subject
                )
            except KeyError:
                self._json({"status": "not_found"}, HTTPStatus.NOT_FOUND)
                return
            except UploadRejected as exc:
                self._json({"status": "rejected", "error": str(exc)}, HTTPStatus.CONFLICT)
                return
            get_audit_ledger().append(
                "upload.approved",
                {
                    "upload_id": metadata["upload_id"],
                    "tenant_id": principal.tenant_id,
                    "actor": principal.subject,
                    "sha256": metadata["sha256"],
                },
            )
            self._json(metadata)
            return
        if path == "/uploads":
            principal = self._require("analyst")
            if principal is None:
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = -1
            if size < 1 or size > UPLOAD_MAX_BODY_BYTES:
                self._json(
                    {"status": "rejected", "error": "upload_size_invalid"},
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
                return
            filename = self.headers.get("X-Filename", "")
            content_type = self.headers.get("Content-Type", "application/octet-stream")
            try:
                metadata = UPLOAD_MANAGER.receive(
                    tenant_id=principal.tenant_id,
                    actor=principal.subject,
                    filename=filename,
                    content_type=content_type,
                    payload=self.rfile.read(size),
                )
            except UploadRejected as exc:
                get_audit_ledger().append(
                    "upload.rejected",
                    {
                        "tenant_id": principal.tenant_id,
                        "actor": principal.subject,
                        "filename_sha256": hashlib.sha256(filename.encode("utf-8")).hexdigest(),
                        "reason": str(exc),
                    },
                )
                self._json({"status": "rejected", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            get_audit_ledger().append(
                "upload.quarantined",
                {
                    "upload_id": metadata["upload_id"],
                    "tenant_id": principal.tenant_id,
                    "actor": principal.subject,
                    "sha256": metadata["sha256"],
                    "bytes": metadata["bytes"],
                    "malware_status": metadata["malware_scan"]["status"],
                },
            )
            self._json(metadata, HTTPStatus.ACCEPTED)
            return
        export_match = re.fullmatch(r"/batch/jobs/([A-Za-z0-9_.-]+)/submission", path)
        if export_match:
            job_id = export_match.group(1)
            principal = self._require("approver")
            if principal is None:
                return
            approval = self.headers.get("X-Kingpro-Approval", "").strip().lower()
            if approval != "user-confirmed":
                self._json(
                    {
                        "status": "approval_required",
                        "approval": approval_contract("export_submission"),
                    },
                    HTTPStatus.FORBIDDEN,
                )
                return
            try:
                report = JOB_MANAGER.export_submission(job_id, principal.tenant_id)
            except KeyError:
                self._json({"status": "not_found", "job_id": job_id}, HTTPStatus.NOT_FOUND)
                return
            except (FileNotFoundError, TypeError, ValueError) as exc:
                self._json({"status": "error", "error": str(exc)}, HTTPStatus.CONFLICT)
                return
            get_audit_ledger().append(
                "submission.export",
                {
                    "job_id": job_id,
                    "approval": "user-confirmed",
                    "archive_sha256": report.get("archive_sha256"),
                    "records": report.get("records"),
                },
            )
            self._json(
                {
                    **report,
                    "approval": approval_contract(
                        "export_submission", approved=True, actor=principal.subject
                    ),
                }
            )
            return
        if path not in {"/ask", "/ask/stream", "/batch/jobs"}:
            self._json({"status": "not_found"}, HTTPStatus.NOT_FOUND)
            return
        principal = self._require("analyst" if path == "/batch/jobs" else "viewer")
        if principal is None:
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = -1
        body_limit = BATCH_MAX_BODY_BYTES if path == "/batch/jobs" else MAX_BODY_BYTES
        if size < 0 or size > body_limit:
            self._json({"status": "error", "error": "request_too_large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            body = json.loads(self.rfile.read(size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json({"status": "error", "error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/batch/jobs":
            items = body.get("items") if isinstance(body, dict) else body
            workers = body.get("workers", 4) if isinstance(body, dict) else 4
            fallback_on_refusal = bool(body.get("fallback_on_refusal", False)) if isinstance(body, dict) else False
            if not isinstance(items, list):
                self._json({"status": "error", "error": "items_must_be_array"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                snapshot = JOB_MANAGER.create(
                    items,
                    workers=int(workers),
                    fallback_on_refusal=fallback_on_refusal,
                    tenant_id=principal.tenant_id,
                    owner_id=principal.subject,
                )
            except (TypeError, ValueError) as exc:
                self._json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json(snapshot, HTTPStatus.ACCEPTED)
            return
        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str):
            self._json({"status": "error", "error": "question_must_be_string"}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/ask/stream":
            self._stream_question(question, principal)
            return
        try:
            result = SERVICE.ask(question)
        except Exception as exc:
            self._json(
                {"status": "error", "grounded": False,
                 "error": f"internal_error: {type(exc).__name__}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        result["principal"] = principal.public()
        self._json(result, HTTPStatus.OK)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[api] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local KINGPRO product API.")
    parser.add_argument("--host", default=os.getenv("KINGPRO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("KINGPRO_PORT", "8080")))
    args = parser.parse_args()
    host = args.host
    port = args.port
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"KINGPRO API: http://{host}:{port}  (GET /health, POST /ask)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
