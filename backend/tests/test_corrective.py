"""Corrective RAG: grade → rewrite → re-retrieve → answer, with a fake LLM."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from app.config import get_settings
from app.rag.agentic.llm import Completion, TurnResult
from app.rag.base import RetrievedContext
from app.rag.corrective.strategy import CorrectiveRAGStrategy


class ScriptedLLM:
    """complete() returns queued grader/rewrite outputs; stream_turn() the answer."""

    model = "claude-sonnet-4-6"

    def __init__(self, completions: list[str], answer: str) -> None:
        self._completions = completions
        self._answer = answer
        self.complete_calls: list[str] = []

    async def complete(self, system, user, model=None) -> Completion:
        self.complete_calls.append(user)
        text = self._completions.pop(0)
        return Completion(text=text, input_tokens=50, output_tokens=20)

    async def stream_turn(self, system, messages, tools) -> AsyncIterator[tuple[str, Any]]:
        self.last_user = messages[-1]["content"]
        for ch in self._answer:
            yield ("delta", ch)
        yield ("final", TurnResult(self._answer, [], [], "end_turn", 200, 80))


class SwitchToolbox:
    """Round 1 vector search returns junk; after rewrite, keyword search returns gold."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict) -> tuple[str, list[RetrievedContext]]:
        self.calls.append((name, args))
        if name == "vector_search":
            c = RetrievedContext("無關內容", "EP100", 100, "2024-01-01", 0.0, 30.0, 0.2, "vector")
        else:  # keyword_search after rewrite
            c = RetrievedContext("輝達財報亮眼", "EP512", 512, "2024-12-07", 750.0, 780.0, 0.9, "keyword")
        return "…", [c]


@pytest.mark.asyncio
async def test_crag_corrects_then_answers():
    # round1 grade: INCORRECT → triggers rewrite; round2 grade: CORRECT with id 0
    llm = ScriptedLLM(
        completions=['{"relevant_ids":[],"verdict":"INCORRECT"}', "輝達 NVDA 財報",
                     '{"relevant_ids":[0],"verdict":"CORRECT"}'],
        answer="根據逐字稿，股癌看好輝達（EP512）。這不是投資建議。",
    )
    tb = SwitchToolbox()
    s = get_settings()
    s.crag_max_rounds = 3
    strat = CorrectiveRAGStrategy(s, toolbox=tb, llm=llm)

    events = [e async for e in strat.answer("股癌怎麼看輝達？")]
    done = events[-1].trace

    assert done["rounds"] == 2 and done["rewrites"] == 1
    assert done["final_verdict"] == "CORRECT"
    # corrected from vector → keyword retriever
    assert [c[0] for c in tb.calls] == ["vector_search", "keyword_search"]
    # only the graded-relevant passage survived
    final_ctx = [e for e in events if e.type == "contexts"][-1].contexts
    assert len(final_ctx) == 1 and final_ctx[0].episode_id == "EP512"
    answer = "".join(e.delta for e in events if e.type == "token")
    assert "輝達" in answer
    assert done["cost_usd"] > 0


@pytest.mark.asyncio
async def test_crag_stops_early_when_first_retrieval_good():
    llm = ScriptedLLM(completions=['{"relevant_ids":[0],"verdict":"CORRECT"}'], answer="答案 EP512。")
    tb = SwitchToolbox()
    s = get_settings()
    strat = CorrectiveRAGStrategy(s, toolbox=tb, llm=llm)
    events = [e async for e in strat.answer("Q")]
    done = events[-1].trace
    assert done["rounds"] == 1 and done["rewrites"] == 0
    assert llm.complete_calls and len(llm.complete_calls) == 1  # graded once, no rewrite
