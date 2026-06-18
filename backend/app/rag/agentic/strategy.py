"""Agentic RAG: Claude drives its own retrieval via a tool-use loop.

Claude decides which tools to call (vector/keyword search, metadata, context
expansion), how many times, and when it has enough to answer — enabling query
rewriting and multi-hop retrieval. We stream the answer tokens to the UI as they
arrive and accumulate every retrieved context + a cost/latency trace.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from app.config import Settings, get_settings
from app.rag.agentic.llm import LLM, TurnResult
from app.rag.agentic.toolbox import TOOL_SCHEMAS, RetrievalToolbox
from app.rag.base import BaseRAGStrategy, Message, RetrievedContext, StreamEvent

SYSTEM_PROMPT = """你是「股癌」podcast 的問答助理。你的知識來自節目逐字稿，必須透過工具檢索後再回答。

規則：
1. 先用 vector_search 做語意檢索；若要找特定股票代號、公司或人名，改用 keyword_search。
2. 視需要多次檢索或改寫查詢；可用 expand_context 取得片段前後文，用 get_episode_metadata 查證集數資訊。
3. 只根據檢索到的逐字稿回答，不要臆測。檢索不到就誠實說明。
4. 一律用繁體中文回答，並在關鍵論點後標註來源，格式如：（EP512 12:30）。
5. 股癌是投資理財節目，內容為個人觀點，回答時提醒這不是投資建議。"""

# $ per 1M tokens (input, output) — for the cost line in the trace.
_PRICING = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
    "gemini-2.5-flash": (0.0, 0.0),  # free tier
}


class AgenticRAGStrategy(BaseRAGStrategy):
    name = "agentic"
    description = "Agentic RAG — Claude orchestrates vector/keyword retrieval via tool use."

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        toolbox: RetrievalToolbox | None = None,
        llm: LLM | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self._toolbox = toolbox or RetrievalToolbox(self._s)
        self._llm = llm  # injected in tests; created lazily otherwise

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            from app.rag.agentic.llm import get_llm

            self._llm = get_llm(self._s)
        return self._llm

    async def answer(
        self,
        query: str,
        history: list[Message] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        t0 = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in (history or [])
        ]
        messages.append({"role": "user", "content": query})

        contexts: dict[tuple[str, float], RetrievedContext] = {}
        in_tok = out_tok = tool_calls = 0
        last_final: TurnResult | None = None

        for _ in range(self._s.agent_max_iterations):
            final: TurnResult | None = None
            async for kind, payload in self.llm.stream_turn(SYSTEM_PROMPT, messages, TOOL_SCHEMAS):
                if kind == "delta":
                    yield StreamEvent(type="token", delta=payload)
                elif kind == "final":
                    final = payload
            assert final is not None
            last_final = final
            in_tok += final.input_tokens
            out_tok += final.output_tokens

            if final.stop_reason != "tool_use" or not final.tool_uses:
                break

            # Append assistant turn (text + tool_use blocks) verbatim, then results.
            messages.append({"role": "assistant", "content": final.raw_content})
            tool_results = []
            for tu in final.tool_uses:
                tool_calls += 1
                result_text, hits = self._toolbox.execute(tu.name, tu.input)
                for h in hits:
                    contexts[(h.episode_id, h.start_s or 0.0)] = h
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": result_text}
                )
            # surface retrievals incrementally
            yield StreamEvent(type="contexts", contexts=_ranked(contexts))
            messages.append({"role": "user", "content": tool_results})
        else:
            # loop exhausted without a natural finish
            yield StreamEvent(type="token", delta="\n\n（已達檢索上限，根據目前資料作答）")

        yield StreamEvent(type="contexts", contexts=_ranked(contexts))
        yield StreamEvent(
            type="done",
            trace={
                "strategy": self.name,
                "model": self.llm.model,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "tokens": {"input": in_tok, "output": out_tok},
                "cost_usd": _cost(self.llm.model, in_tok, out_tok),
                "tool_calls": tool_calls,
                "stop_reason": last_final.stop_reason if last_final else None,
                "n_contexts": len(contexts),
            },
        )


def _ranked(contexts: dict[tuple[str, float], RetrievedContext]) -> list[RetrievedContext]:
    return sorted(contexts.values(), key=lambda c: c.score, reverse=True)


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = _PRICING.get(model, (0.0, 0.0))
    return round(in_tok / 1e6 * pin + out_tok / 1e6 * pout, 6)
