"""FastAPI app shell.

Intentionally thin and strategy-agnostic. The only endpoints are:
  - GET  /health
  - GET  /strategies   -> drives the frontend dropdown
  - POST /chat         -> Server-Sent Events stream of StreamEvent dicts

These do not change as RAG strategies are added or iterated.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag.base import Message
from app.rag.registry import build_default_registry

logger = logging.getLogger("gooaye.api")
settings = get_settings()
registry = build_default_registry(settings)

if not settings.api_auth_token:
    logger.warning(
        "API_AUTH_TOKEN is empty — /chat and /strategies are UNAUTHENTICATED. "
        "Set it before exposing this service on a network."
    )

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_auth(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """If a token is configured, require it; otherwise open (dev)."""
    token = settings.api_auth_token
    if not token:
        return
    provided = None
    if authorization and authorization.startswith("Bearer "):
        provided = authorization[len("Bearer ") :]
    elif x_api_key:
        provided = x_api_key
    if provided != token:
        raise HTTPException(status_code=401, detail="Unauthorized")


_hits: dict[str, deque[float]] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    """Simple in-memory per-IP fixed-window limiter (0 = disabled)."""
    rpm = settings.rate_limit_per_min
    if rpm <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    dq = _hits[ip]
    while dq and dq[0] < now - 60:
        dq.popleft()
    if len(dq) >= rpm:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    dq.append(now)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    strategy: str
    history: list[ChatTurn] = Field(default_factory=list)
    filters: dict[str, Any] | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/strategies", dependencies=[Depends(rate_limit), Depends(require_auth)])
async def strategies() -> dict[str, Any]:
    """List available strategies. ``default`` is the first enabled one."""
    described = registry.describe()
    return {"strategies": described, "default": described[0]["name"] if described else None}


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/chat", dependencies=[Depends(rate_limit), Depends(require_auth)])
async def chat(req: ChatRequest) -> StreamingResponse:
    try:
        strategy = registry.get(req.strategy)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown strategy: {req.strategy!r}")

    history = [Message(role=t.role, content=t.content) for t in req.history]

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in strategy.answer(req.query, history=history, filters=req.filters):
                yield _sse(event.as_dict())
        except Exception as exc:  # surface errors to the client instead of hanging
            yield _sse({"type": "error", "delta": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
