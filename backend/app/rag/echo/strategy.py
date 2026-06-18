"""EchoStrategy — a zero-dependency stub that proves the pluggable contract.

It performs no real retrieval or generation: it emits one fake context, streams
back the user's query token-by-token, then a trace. Its only job is to verify
that ``/strategies`` → dropdown → ``/chat`` → SSE stream works before any real
RAG exists. Delete or keep as a smoke-test once real strategies land.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from app.rag.base import BaseRAGStrategy, Message, RetrievedContext, StreamEvent


class EchoStrategy(BaseRAGStrategy):
    name = "echo"
    description = "Stub strategy (no retrieval) — echoes the query to validate the pipeline."

    async def answer(
        self,
        query: str,
        history: list[Message] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        t0 = time.perf_counter()

        # 1) Pretend we retrieved something.
        yield StreamEvent(
            type="contexts",
            contexts=[
                RetrievedContext(
                    text=f"(stub context for: {query})",
                    episode_id="EP000",
                    ep_number=0,
                    publish_date=None,
                    score=1.0,
                    source="stub",
                )
            ],
        )

        # 2) Stream a canned answer token-by-token (simulates LLM streaming).
        reply = f"echo｜你問了：「{query}」。這是 EchoStrategy 的測試回覆。"
        for ch in reply:
            yield StreamEvent(type="token", delta=ch)
            await asyncio.sleep(0.005)

        # 3) Final trace — same shape real strategies will emit for eval.
        yield StreamEvent(
            type="done",
            trace={
                "strategy": self.name,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "tokens": {"input": 0, "output": len(reply)},
                "cost_usd": 0.0,
                "tool_calls": 0,
                "filters": filters or {},
            },
        )
