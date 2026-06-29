"""Hybrid + Rerank RAG: fuse vector + keyword retrieval, then cross-encoder rerank.

Retrieval (all free, local — no LLM until the answer step):
  1. pull top candidates from BOTH retrievers — vector (semantic) and keyword (BM25);
  2. fuse the two ranked lists with Reciprocal-Rank Fusion (RRF) at the chunk level;
  3. re-order the fused candidates with a local cross-encoder reranker and keep top-k;
  4. generate the answer from only those reranked passages.

Why this exists: the retrieval scorecard showed hybrid beats either retriever alone,
and a cross-encoder is the cheapest precision win on top of that — it judges each
(query, passage) pair jointly, fixing cases where the bi-encoder ranks a loosely
related chunk too high. Reuses the shared retrieval toolbox and the streaming LLM seam,
so it's testable with fakes and emits the same trace shape as the other strategies.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, AsyncIterator

from app.config import Settings, get_settings
from app.rag.agentic.llm import LLM
from app.rag.agentic.strategy import _cost
from app.rag.agentic.toolbox import RetrievalToolbox
from app.rag.base import BaseRAGStrategy, Message, RetrievedContext, StreamEvent

GEN_SYSTEM = """你是「股癌」podcast 的問答助理。以下是經過混合檢索與重排後、最相關的逐字稿片段。
只根據這些片段，用繁體中文回答問題，並在關鍵論點後標註來源（如 EP512）。
若片段不足以回答就誠實說明。股癌為投資理財節目，內容屬個人觀點。"""


def _rrf_fuse(
    lists: list[list[RetrievedContext]], k: int = 60
) -> list[RetrievedContext]:
    """Reciprocal-Rank Fusion of several ranked context lists, deduped by (episode,
    start time). A chunk appearing high in both lists rises; order is by fused score."""
    score: dict[tuple[str, float], float] = defaultdict(float)
    keep: dict[tuple[str, float], RetrievedContext] = {}
    for lst in lists:
        for rank, c in enumerate(lst, 1):
            key = (c.episode_id, round(c.start_s or 0.0, 1))
            score[key] += 1.0 / (k + rank)
            keep.setdefault(key, c)
    return [keep[key] for key in sorted(score, key=lambda kk: score[kk], reverse=True)]


class HybridRerankRAGStrategy(BaseRAGStrategy):
    name = "hybrid_rerank"
    description = "Hybrid RAG — fuse vector + keyword (RRF), local cross-encoder rerank, then answer."

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        toolbox: RetrievalToolbox | None = None,
        reranker=None,
        llm: LLM | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self._toolbox = toolbox or RetrievalToolbox(self._s)
        self._reranker = reranker
        self._llm = llm

    @property
    def reranker(self):
        if self._reranker is None:
            from app.retrieval.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(self._s)
        return self._reranker

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            from app.rag.agentic.llm import get_llm

            self._llm = get_llm(self._s)
        return self._llm

    def _retrieve(self, query: str) -> tuple[list[RetrievedContext], int]:
        """Hybrid-fuse then rerank. Returns (reranked top-k, n_candidates_fused)."""
        k = self._s.hybrid_k
        _, vhits = self._toolbox.execute("vector_search", {"query": query, "k": k})
        _, khits = self._toolbox.execute("keyword_search", {"query": query, "k": k})
        fused = _rrf_fuse([vhits, khits])
        reranked = self.reranker.rerank(query, fused, top_k=self._s.rerank_top_k)
        return reranked, len(fused)

    async def answer(
        self,
        query: str,
        history: list[Message] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        t0 = time.perf_counter()
        contexts, n_candidates = self._retrieve(query)
        yield StreamEvent(type="contexts", contexts=contexts)

        ctx_block = "\n\n".join(
            f"[{c.episode_id} | {c.publish_date}] {c.text}" for c in contexts
        ) or "（找不到相關片段）"
        user = f"問題：{query}\n\n相關片段：\n{ctx_block}"
        messages = [{"role": m.role, "content": m.content} for m in (history or [])]
        messages.append({"role": "user", "content": user})

        in_tok = out_tok = 0
        stop_reason = "end_turn"
        async for kind, payload in self.llm.stream_turn(GEN_SYSTEM, messages, []):
            if kind == "delta":
                yield StreamEvent(type="token", delta=payload)
            elif kind == "final":
                in_tok, out_tok = payload.input_tokens, payload.output_tokens
                stop_reason = payload.stop_reason

        yield StreamEvent(
            type="done",
            trace={
                "strategy": self.name,
                "model": self.llm.model,
                "rerank_model": self._s.rerank_model,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "tokens": {"input": in_tok, "output": out_tok},
                "cost_usd": _cost(self.llm.model, in_tok, out_tok),
                "tool_calls": 0,
                "n_candidates": n_candidates,
                "n_contexts": len(contexts),
                "stop_reason": stop_reason,
            },
        )

    async def retrieve(self, query, k=8, filters=None):
        contexts, _ = self._retrieve(query)
        return contexts[:k]
