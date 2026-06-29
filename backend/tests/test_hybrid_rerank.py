"""Hybrid + Rerank RAG: fuse vector + keyword, rerank, answer — with fakes."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from app.rag.agentic.llm import TurnResult
from app.rag.base import RetrievedContext
from app.rag.hybrid_rerank.strategy import HybridRerankRAGStrategy, _rrf_fuse


class FakeToolbox:
    """vector_search and keyword_search return overlapping-but-different hits."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, args: dict) -> tuple[str, list[RetrievedContext]]:
        self.calls.append((name, args))
        if name == "vector_search":
            return "…", [
                RetrievedContext("輝達財報", "EP512", 512, "2024-12-07", 750.0, 780.0, 0.7, "vector"),
                RetrievedContext("無關內容", "EP100", 100, "2024-01-01", 0.0, 30.0, 0.3, "vector"),
            ]
        return "…", [  # keyword_search — EP512 also here (should fuse, not duplicate)
            RetrievedContext("輝達財報", "EP512", 512, "2024-12-07", 750.0, 780.0, 0.6, "keyword"),
            RetrievedContext("台積電擴廠", "EP511", 511, "2024-12-01", 120.0, 150.0, 0.5, "keyword"),
        ]


class FakeReranker:
    """Scores by text length as a stand-in; records what it was asked to rank."""

    def __init__(self) -> None:
        self.seen: list[RetrievedContext] = []

    def rerank(self, query: str, contexts: list[RetrievedContext], top_k: int):
        self.seen = contexts
        ranked = sorted(contexts, key=lambda c: len(c.text), reverse=True)
        for i, c in enumerate(ranked):
            c.score = 1.0 - i * 0.1
            c.source = "rerank"
        return ranked[:top_k]


class ScriptedLLM:
    model = "llama-3.3-70b-versatile"
    grader_model = "llama-3.1-8b-instant"

    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def stream_turn(self, system, messages, tools) -> AsyncIterator[tuple[str, Any]]:
        self.last_user = messages[-1]["content"]
        for ch in self._answer:
            yield ("delta", ch)
        yield ("final", TurnResult(self._answer, [], [], "end_turn", 200, 80))


def test_rrf_fuse_dedups_by_episode_and_time():
    a = [RetrievedContext("x", "EP512", 512, None, 750.0, 780.0, 0.7, "vector")]
    b = [RetrievedContext("x", "EP512", 512, None, 750.0, 780.0, 0.6, "keyword")]
    fused = _rrf_fuse([a, b])
    assert len(fused) == 1  # same (episode, start) collapses to one


@pytest.mark.asyncio
async def test_hybrid_rerank_fuses_reranks_and_answers():
    tb, rr = FakeToolbox(), FakeReranker()
    llm = ScriptedLLM("根據逐字稿，股癌看好輝達（EP512）。")
    strat = HybridRerankRAGStrategy(toolbox=tb, reranker=rr, llm=llm)

    events = [ev async for ev in strat.answer("輝達")]

    # both retrievers were queried
    names = {name for name, _ in tb.calls}
    assert names == {"vector_search", "keyword_search"}

    # EP512 appeared in both lists → fused to a single candidate (3 unique, not 4)
    assert len(rr.seen) == 3

    contexts = next(ev.contexts for ev in events if ev.type == "contexts")
    assert contexts and all(c.source == "rerank" for c in contexts)

    done = next(ev for ev in events if ev.type == "done")
    assert done.trace["strategy"] == "hybrid_rerank"
    assert done.trace["n_candidates"] == 3
    assert done.trace["n_contexts"] == len(contexts)
    assert "rerank_model" in done.trace

    answer = "".join(ev.delta for ev in events if ev.type == "token")
    assert "輝達" in answer
