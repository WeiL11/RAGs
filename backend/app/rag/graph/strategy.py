"""Graph RAG: retrieve via the knowledge graph, then synthesize with Claude.

Query flow (retrieval is deterministic — no LLM until synthesis):
  1. link query terms to graph entities;
  2. expand to 1-hop neighbors;
  3. collect those entities' mention chunks (provenance) + relation triples;
  4. Claude synthesizes an answer grounded in that subgraph, streaming.

Reuses the same streaming LLM seam as Agentic RAG, so it's testable with a fake LLM
and emits the same trace shape for head-to-head eval.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from app.config import Settings, get_settings
from app.rag.agentic.llm import LLM
from app.rag.agentic.strategy import _PRICING, _cost
from app.rag.base import BaseRAGStrategy, Message, RetrievedContext, StreamEvent

SYSTEM_PROMPT = """你是「股癌」podcast 的問答助理。以下提供從知識圖譜檢索到的相關片段與實體關係。
請只根據這些內容，用繁體中文回答問題，並在關鍵論點後標註來源（如 EP512）。
若資訊不足就誠實說明。股癌為投資理財節目，內容屬個人觀點。"""


def _tokenize(query: str) -> list[str]:
    try:
        import jieba

        return [t for t in jieba.lcut(query) if len(t.strip()) > 1]
    except Exception:
        return query.split()


class GraphRAGStrategy(BaseRAGStrategy):
    name = "graph"
    description = "Graph RAG — entity-relation graph traversal + Claude synthesis."

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        graph=None,
        llm: LLM | None = None,
        hops: int = 1,
    ) -> None:
        self._s = settings or get_settings()
        self._graph = graph
        self._llm = llm
        self._hops = hops

    @property
    def graph(self):
        if self._graph is None:
            from app.retrieval.graph_store import GraphStore

            self._graph = GraphStore(self._s.graph_path).load()
        return self._graph

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            from app.rag.agentic.llm import get_llm

            self._llm = get_llm(self._s)
        return self._llm

    def _retrieve(self, query: str) -> tuple[list[RetrievedContext], list[str]]:
        terms = _tokenize(query)
        seeds = self.graph.link_entities(query, terms)
        expanded = set(seeds)
        for s in seeds:
            expanded |= self.graph.neighbors(s, hops=self._hops)
        contexts = self.graph.mentions_as_contexts(expanded)
        triples = self.graph.relation_triples(expanded)
        return contexts, triples

    async def answer(
        self,
        query: str,
        history: list[Message] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        t0 = time.perf_counter()
        contexts, triples = self._retrieve(query)
        yield StreamEvent(type="contexts", contexts=contexts)

        # Build the grounded prompt from the retrieved subgraph.
        ctx_block = "\n\n".join(
            f"[{c.episode_id} | {c.publish_date}] {c.text}" for c in contexts
        ) or "（圖譜中找不到相關片段）"
        rel_block = "\n".join(triples) or "（無相關實體關係）"
        user = f"問題：{query}\n\n相關逐字稿片段：\n{ctx_block}\n\n實體關係：\n{rel_block}"
        messages = [{"role": m.role, "content": m.content} for m in (history or [])]
        messages.append({"role": "user", "content": user})

        in_tok = out_tok = 0
        stop_reason = "end_turn"
        async for kind, payload in self.llm.stream_turn(SYSTEM_PROMPT, messages, []):
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
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "tokens": {"input": in_tok, "output": out_tok},
                "cost_usd": _cost(self.llm.model, in_tok, out_tok),
                "tool_calls": 0,
                "stop_reason": stop_reason,
                "n_contexts": len(contexts),
                "n_relations": len(triples),
            },
        )

    async def retrieve(self, query, k=8, filters=None):
        contexts, _ = self._retrieve(query)
        return contexts[:k]


# keep a reference so linters don't flag the imported pricing table as unused
_ = _PRICING
