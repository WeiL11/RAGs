"""Corrective RAG (CRAG): grade retrieved passages, self-correct, then answer.

Loop per query:
  1. retrieve top-k (vector, then keyword on retries);
  2. a cheap grader LLM judges relevance → verdict CORRECT / AMBIGUOUS / INCORRECT;
  3. if INCORRECT (nothing relevant), rewrite the query and retrieve again;
  4. stop when relevant passages are found or rounds run out;
  5. generate the answer from only the passages that passed grading.

The classic CRAG falls back to web search; our corpus is closed, so "correction"
means query rewriting + switching the retriever. Reuses the shared retrieval
toolbox (free, local) and the streaming LLM seam (testable with a fake).
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from app.config import Settings, get_settings
from app.rag.agentic.llm import LLM
from app.rag.agentic.strategy import _cost
from app.rag.agentic.toolbox import RetrievalToolbox
from app.rag.base import BaseRAGStrategy, Message, RetrievedContext, StreamEvent

GEN_SYSTEM = """你是「股癌」podcast 的問答助理。以下是經過相關性篩選的逐字稿片段。
只根據這些片段，用繁體中文回答問題，並在關鍵論點後標註來源（如 EP512）。
若片段不足以回答就誠實說明。股癌為投資理財節目，內容屬個人觀點，請提醒這不是投資建議。"""

GRADE_SYSTEM = """你是檢索結果的相關性評分器。判斷每段逐字稿能否協助回答使用者的問題。
只輸出 JSON：{"relevant_ids":[相關片段的編號], "verdict":"CORRECT|AMBIGUOUS|INCORRECT"}
CORRECT=至少一段高度相關；AMBIGUOUS=部分沾邊；INCORRECT=全部不相關。不要輸出其他文字。"""

REWRITE_SYSTEM = """你會改寫檢索查詢，使其更可能在投資 podcast 逐字稿中命中相關內容。
可加入同義詞、股票/公司全名、相關概念。只輸出改寫後的查詢字串，不要解釋。"""


class CorrectiveRAGStrategy(BaseRAGStrategy):
    name = "corrective"
    description = "Corrective RAG — grade retrieved passages, rewrite & re-retrieve when weak, then answer."

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        toolbox: RetrievalToolbox | None = None,
        llm: LLM | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self._toolbox = toolbox or RetrievalToolbox(self._s)
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            from app.rag.agentic.llm import get_llm

            self._llm = get_llm(self._s)
        return self._llm

    async def _grade(self, query: str, hits: list[RetrievedContext]) -> tuple[str, list[int], int, int]:
        if not hits:
            return "INCORRECT", [], 0, 0
        listing = "\n".join(f"[{i}] {h.text[:200]}" for i, h in enumerate(hits))
        comp = await self.llm.complete(
            GRADE_SYSTEM, f"問題：{query}\n\n片段：\n{listing}", model=self.llm.grader_model
        )
        verdict, ids = "AMBIGUOUS", list(range(len(hits)))
        try:
            data = json.loads(comp.text[comp.text.find("{") : comp.text.rfind("}") + 1])
            verdict = str(data.get("verdict", "AMBIGUOUS")).upper()
            ids = [int(i) for i in data.get("relevant_ids", []) if 0 <= int(i) < len(hits)]
        except Exception:
            pass  # fail-open: keep all hits if the grader output is unparseable
        return verdict, ids, comp.input_tokens, comp.output_tokens

    async def _rewrite(self, original: str, current: str) -> tuple[str, int, int]:
        comp = await self.llm.complete(
            REWRITE_SYSTEM, f"原始問題：{original}\n目前查詢：{current}", model=self.llm.grader_model
        )
        new_q = comp.text.strip().splitlines()[0].strip() if comp.text.strip() else current
        return (new_q or current), comp.input_tokens, comp.output_tokens

    async def answer(
        self,
        query: str,
        history: list[Message] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        t0 = time.perf_counter()
        kept: dict[tuple[str, float], RetrievedContext] = {}
        q, mode = query, "vector"
        rounds = rewrites = grade_in = grade_out = 0
        verdict = "INCORRECT"

        for r in range(self._s.crag_max_rounds):
            rounds += 1
            _, hits = self._toolbox.execute(f"{mode}_search", {"query": q, "k": self._s.crag_k})
            verdict, rel_ids, gin, gout = await self._grade(query, hits)
            grade_in += gin
            grade_out += gout
            for i in rel_ids:
                h = hits[i]
                kept[(h.episode_id, h.start_s or 0.0)] = h
            yield StreamEvent(type="contexts", contexts=_ranked(kept))

            if verdict != "INCORRECT" and kept:
                break
            if r < self._s.crag_max_rounds - 1:  # corrective action
                q, rin, rout = await self._rewrite(query, q)
                grade_in += rin
                grade_out += rout
                rewrites += 1
                mode = "keyword" if mode == "vector" else "vector"  # diversify retriever

        contexts = _ranked(kept)
        yield StreamEvent(type="contexts", contexts=contexts)

        ctx_block = "\n\n".join(
            f"[{c.episode_id} | {c.publish_date}] {c.text}" for c in contexts
        ) or "（找不到相關片段）"
        user = f"問題：{query}\n\n相關片段：\n{ctx_block}"
        messages = [{"role": m.role, "content": m.content} for m in (history or [])]
        messages.append({"role": "user", "content": user})

        gen_in = gen_out = 0
        async for kind, payload in self.llm.stream_turn(GEN_SYSTEM, messages, []):
            if kind == "delta":
                yield StreamEvent(type="token", delta=payload)
            elif kind == "final":
                gen_in, gen_out = payload.input_tokens, payload.output_tokens

        cost = _cost(self.llm.grader_model, grade_in, grade_out) + _cost(
            self.llm.model, gen_in, gen_out
        )
        yield StreamEvent(
            type="done",
            trace={
                "strategy": self.name,
                "model": self.llm.model,
                "grader_model": self.llm.grader_model,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "tokens": {"input": grade_in + gen_in, "output": grade_out + gen_out},
                "cost_usd": round(cost, 6),
                "tool_calls": 0,
                "rounds": rounds,
                "rewrites": rewrites,
                "final_verdict": verdict,
                "n_contexts": len(contexts),
            },
        )

    async def retrieve(self, query, k=8, filters=None):
        _, hits = self._toolbox.execute("vector_search", {"query": query, "k": k})
        return hits[:k]


def _ranked(kept: dict[tuple[str, float], RetrievedContext]) -> list[RetrievedContext]:
    return sorted(kept.values(), key=lambda c: c.score, reverse=True)
