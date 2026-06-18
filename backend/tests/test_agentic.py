"""Agentic RAG loop tested end-to-end with a scripted fake LLM + stub toolbox."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from app.config import get_settings
from app.rag.agentic.llm import ToolUse, TurnResult
from app.rag.agentic.strategy import AgenticRAGStrategy
from app.rag.base import RetrievedContext


class FakeLLM:
    """Replays scripted turns. Each turn: (text, [(tool_name, input)] or None)."""

    model = "claude-opus-4-8"

    def __init__(self, script: list[tuple[str, list[tuple[str, dict]] | None]]) -> None:
        self._script = script
        self._i = 0

    async def stream_turn(self, system, messages, tools) -> AsyncIterator[tuple[str, Any]]:
        text, tool_calls = self._script[self._i]
        self._i += 1
        for ch in text:  # stream char-by-char like the real text_stream
            yield ("delta", ch)
        tool_uses = [
            ToolUse(id=f"tu_{i}", name=n, input=inp) for i, (n, inp) in enumerate(tool_calls or [])
        ]
        raw = [{"type": "text", "text": text}] + [
            {"type": "tool_use", "id": t.id, "name": t.name, "input": t.input} for t in tool_uses
        ]
        yield (
            "final",
            TurnResult(
                text=text,
                tool_uses=tool_uses,
                raw_content=raw,
                stop_reason="tool_use" if tool_uses else "end_turn",
                input_tokens=100,
                output_tokens=50,
            ),
        )


class StubToolbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict) -> tuple[str, list[RetrievedContext]]:
        self.calls.append((name, args))
        ctx = RetrievedContext(
            text="輝達財報亮眼，資料中心需求強勁。",
            episode_id="EP512",
            ep_number=512,
            publish_date="2024-12-07",
            start_s=750.0,
            end_s=780.0,
            score=0.91,
            source="vector",
        )
        return "[EP512 @ 12:30] 輝達財報亮眼…", [ctx]


@pytest.mark.asyncio
async def test_agentic_tool_loop_streams_and_traces():
    script = [
        ("我先查一下。", [("vector_search", {"query": "輝達 財報"})]),  # turn 1: tool call
        ("根據逐字稿，股癌認為輝達財報亮眼（EP512 12:30）。這不是投資建議。", None),  # turn 2: answer
    ]
    toolbox = StubToolbox()
    strat = AgenticRAGStrategy(get_settings(), toolbox=toolbox, llm=FakeLLM(script))

    events = [e async for e in strat.answer("股癌怎麼看輝達？")]

    types = [e.type for e in events]
    assert types[0] == "token"  # streamed before tools
    assert "contexts" in types
    assert types[-1] == "done"

    answer = "".join(e.delta for e in events if e.type == "token")
    assert "輝達" in answer and "EP512" in answer

    done = events[-1]
    assert done.trace["tool_calls"] == 1
    assert done.trace["tokens"] == {"input": 200, "output": 100}
    assert done.trace["cost_usd"] > 0
    assert done.trace["n_contexts"] == 1
    assert toolbox.calls == [("vector_search", {"query": "輝達 財報"})]

    final_contexts = [e for e in events if e.type == "contexts"][-1].contexts
    assert final_contexts[0].episode_id == "EP512"


@pytest.mark.asyncio
async def test_agentic_respects_iteration_cap():
    # LLM that always asks for another tool call → must stop at the cap, not hang.
    s = get_settings()
    s.agent_max_iterations = 3
    forever = [("再查一次。", [("vector_search", {"query": "x"})])] * 10
    strat = AgenticRAGStrategy(s, toolbox=StubToolbox(), llm=FakeLLM(forever))
    events = [e async for e in strat.answer("test")]
    assert events[-1].type == "done"
    assert events[-1].trace["tool_calls"] == 3  # capped
