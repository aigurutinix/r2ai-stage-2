"""Small HTTP surface for the presenter-facing ViFinQA demo."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


MAX_QUESTION_LENGTH = int(os.getenv("DEMO_MAX_QUESTION_LENGTH", "2000"))
STATIC_DIR = Path(__file__).with_name("static")


class ChatRequest(BaseModel):
    question: str


class _UnavailablePipeline:
    """Keep the UI available when live inputs or model endpoints are missing."""

    def __init__(self, error: Exception):
        self.error = error

    async def health(self) -> dict[str, Any]:
        raise RuntimeError("pipeline unavailable") from self.error

    async def run(self, question: str, request_id: str) -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "error",
            "stage": "startup",
            "status": "failed",
            "message": "The live pipeline is unavailable; check demo configuration.",
            "data": {},
        }

    def evidence(self, request_id: str, citation_id: str) -> None:
        return None


def _profile() -> dict[str, str]:
    mode = os.getenv("DEMO_MODE", "remote").strip().lower()
    if mode == "fixture":
        return {"mode": "fixture", "label": "FIXTURE DEMO"}
    return {"mode": mode or "remote", "label": "REMOTE GPU"}


def _profile_for(result: dict[str, Any]) -> dict[str, str]:
    mode = result.get("mode") or result.get("profile", "")
    if isinstance(mode, dict):
        mode = mode.get("mode", "")
    mode = str(mode).strip().lower()
    if mode == "fixture":
        return {"mode": "fixture", "label": "FIXTURE DEMO"}
    return _profile()


def _build_pipeline() -> Any:
    from vifinqa.demo_pipeline import DemoPipeline

    return DemoPipeline.from_env()


def _event(event: Any, request_id: str) -> dict[str, Any]:
    if not isinstance(event, dict):
        return {
            "type": "error",
            "stage": "server",
            "status": "error",
            "message": "Pipeline returned an invalid event.",
            "data": {},
            "request_id": request_id,
        }
    return {
        "type": event.get("type", "stage"),
        "stage": event.get("stage", "pipeline"),
        "status": event.get("status", "ok"),
        "message": event.get("message", ""),
        "data": event.get("data", {}),
        "request_id": request_id,
    }


def _line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def create_app(pipeline: Any | None = None) -> FastAPI:
    pipeline_holder = [pipeline]
    active_query = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if pipeline_holder[0] is None:
            try:
                pipeline_holder[0] = _build_pipeline()
            except Exception as error:
                pipeline_holder[0] = _UnavailablePipeline(error)
        yield

    app = FastAPI(title="ViFinQA demo", lifespan=lifespan)

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        # A restart can serve an asset while it is still being written, and a
        # browser that caches that empty response shows a blank page until it is
        # hard-refreshed. Never cache this demo's own files.
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    if STATIC_DIR.is_dir():
        app.mount("/demo/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False, response_model=None)
    async def index() -> FileResponse | PlainTextResponse:
        index_file = STATIC_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return PlainTextResponse("Demo UI is not installed; use /api/health.")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        try:
            details = await pipeline_holder[0].health()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="pipeline unavailable") from exc
        result = details if isinstance(details, dict) else {"data": details}
        result = {**result, "profile": _profile_for(result)}
        result.setdefault("status", "ok")
        return result

    @app.get("/api/evidence/{request_id}/{citation_id}")
    async def evidence(request_id: str, citation_id: str) -> dict[str, Any]:
        try:
            result = pipeline_holder[0].evidence(request_id.strip(), citation_id.strip())
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            raise HTTPException(status_code=503, detail="evidence unavailable") from exc
        if result is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        return result

    @app.get("/api/table/{request_id}/{table_id:path}")
    async def table(request_id: str, table_id: str) -> dict[str, Any]:
        try:
            result = pipeline_holder[0].table_preview(request_id.strip(), table_id.strip())
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            raise HTTPException(status_code=503, detail="table unavailable") from exc
        if result is None:
            raise HTTPException(status_code=404, detail="table not found")
        return result

    @app.post("/api/chat")
    async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
        question = payload.question.strip()
        if not question or len(question) > MAX_QUESTION_LENGTH:
            raise HTTPException(
                status_code=422,
                detail=f"question must contain 1-{MAX_QUESTION_LENGTH} characters",
            )
        if active_query.locked():
            raise HTTPException(status_code=409, detail="another query is already running")
        await active_query.acquire()
        request_id = str(uuid4())

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for pipeline_event in pipeline_holder[0].run(question, request_id):
                    if await request.is_disconnected():
                        return
                    yield _line(_event(pipeline_event, request_id))
            except asyncio.CancelledError:
                raise
            except Exception:
                if not await request.is_disconnected():
                    yield _line(_event({
                        "type": "error",
                        "stage": "pipeline",
                        "status": "error",
                        "message": "The query failed.",
                        "data": {},
                    }, request_id))
            finally:
                active_query.release()

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store"},
        )

    return app


app = create_app()
