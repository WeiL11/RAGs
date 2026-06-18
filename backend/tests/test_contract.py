"""Contract smoke tests — these must keep passing as strategies are added."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.echo.strategy import EchoStrategy

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_strategies_lists_echo():
    body = client.get("/strategies").json()
    names = [s["name"] for s in body["strategies"]]
    assert "echo" in names
    assert body["default"] == "echo"


def test_chat_streams_sse():
    resp = client.post("/chat", json={"query": "測試", "strategy": "echo"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "contexts" in text and "token" in text and "done" in text


def test_chat_unknown_strategy_404():
    resp = client.post("/chat", json={"query": "x", "strategy": "nope"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_echo_event_sequence():
    events = [e async for e in EchoStrategy().answer("hi")]
    assert events[0].type == "contexts"
    assert events[-1].type == "done"
    assert "latency_ms" in events[-1].trace
    # retrieve() default impl should surface the stub context
    ctx = await EchoStrategy().retrieve("hi")
    assert ctx and ctx[0].source == "stub"
