"""Small contract tests for the presenter-facing API."""

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from demo.app import create_app


class FixturePipeline:
    async def run(self, question: str, request_id: str):
        yield {
            "type": "stage",
            "stage": "retrieve",
            "status": "ok",
            "message": question,
            "data": {"profile": "FIXTURE DEMO"},
        }
        yield {
            "type": "result",
            "stage": "answer",
            "status": "ok",
            "message": "done",
            "data": {"answer": 42, "citations": ["c1"]},
        }

    async def health(self):
        return {"status": "ok", "profile": "fixture"}

    async def evidence(self, request_id: str, citation_id: str):
        return {"request_id": request_id, "citation_id": citation_id, "value": 42}

    def table_preview(self, request_id: str, table_id: str):
        if table_id == "missing|1":
            return None
        return {"request_id": request_id, "table_id": table_id, "headers": ["a"], "rows": [["1"]]}


class FailingPipeline(FixturePipeline):
    async def run(self, question: str, request_id: str):
        raise RuntimeError("test failure")
        yield  # Keep this an async generator.


class BlockingPipeline(FixturePipeline):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, question: str, request_id: str):
        self.started.set()
        await self.release.wait()
        yield {"type": "result", "stage": "answer", "status": "ok", "message": "done", "data": {}}


def run(coro):
    return asyncio.run(coro)


async def post(app, payload):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/chat", json=payload)


def test_chat_stream_attaches_request_id_and_fixture_profile():
    response = run(post(create_app(FixturePipeline()), {"question": "fresh question"}))

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["stage", "result"]
    assert len({event["request_id"] for event in events}) == 1
    assert events[0]["data"]["profile"] == "FIXTURE DEMO"


def test_chat_rejects_blank_and_oversized_questions():
    app = create_app(FixturePipeline())
    assert run(post(app, {"question": "   "})).status_code == 422
    assert run(post(app, {"question": "x" * 2001})).status_code == 422


def test_only_one_query_runs_at_a_time():
    async def check():
        pipeline = BlockingPipeline()
        app = create_app(pipeline)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post("/api/chat", json={"question": "first"}))
            await pipeline.started.wait()
            second = await client.post("/api/chat", json={"question": "second"})
            pipeline.release.set()
            await first
        return second

    assert run(check()).status_code == 409


def test_pipeline_error_is_streamed_as_safe_error_event():
    response = run(post(create_app(FailingPipeline()), {"question": "question"}))

    assert response.status_code == 200
    event = json.loads(response.text)
    assert event["type"] == "error"
    assert event["status"] == "error"
    assert "test failure" not in event["message"]


def test_fixture_evidence_is_request_scoped():
    async def check():
        app = create_app(FixturePipeline())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/evidence/r1/c1")

    response = run(check())
    assert response.status_code == 200
    assert response.json() == {"request_id": "r1", "citation_id": "c1", "value": 42}


def test_table_preview_is_served_by_request_and_table_id():
    async def check():
        app = create_app(FixturePipeline())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/table/r1/VJC_..._2018|1179")

    response = run(check())
    assert response.status_code == 200
    assert response.json()["headers"] == ["a"]


def test_table_preview_404s_for_an_unknown_table():
    async def check():
        app = create_app(FixturePipeline())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/table/r1/missing|1")

    response = run(check())
    assert response.status_code == 404


def test_health_profile_comes_from_pipeline():
    async def check():
        app = create_app(FixturePipeline())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/health")

    response = run(check())
    assert response.status_code == 200
    assert response.json()["profile"] == {"mode": "fixture", "label": "FIXTURE DEMO"}
