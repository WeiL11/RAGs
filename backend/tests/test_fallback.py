"""FallbackLLM: primary → secondary ('option B') on request-time error."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from app.config import get_settings
from app.rag.agentic.llm import Completion, FallbackLLM, TurnResult, get_llm


class _LLM:
    def __init__(self, name, fail=False):
        self.model = f"{name}-model"
        self.grader_model = f"{name}-grader"
        self._name = name
        self._fail = fail

    async def stream_turn(self, system, messages, tools) -> AsyncIterator[tuple[str, Any]]:
        if self._fail:
            raise RuntimeError("403 access denied")
        yield ("delta", self._name)
        yield ("final", TurnResult(self._name, [], [], "end_turn", 1, 1))

    async def complete(self, system, user, model=None) -> Completion:
        if self._fail:
            raise RuntimeError("429 quota")
        return Completion(text=f"{self._name}:{model}", input_tokens=1, output_tokens=1)


@pytest.mark.asyncio
async def test_stream_falls_back_when_primary_errors():
    fb = FallbackLLM(_LLM("gemini", fail=True), _LLM("groq"))
    out = [ev async for ev in fb.stream_turn("s", [{"role": "user", "content": "hi"}], [])]
    assert ("delta", "groq") in out  # secondary served the whole turn
    assert out[-1][0] == "final"


@pytest.mark.asyncio
async def test_stream_uses_primary_when_ok():
    fb = FallbackLLM(_LLM("gemini"), _LLM("groq"))
    out = [ev async for ev in fb.stream_turn("s", [{"role": "user", "content": "hi"}], [])]
    assert ("delta", "gemini") in out and ("delta", "groq") not in out


@pytest.mark.asyncio
async def test_complete_falls_back_with_secondary_model():
    fb = FallbackLLM(_LLM("gemini", fail=True), _LLM("groq"))
    # primary grader model must NOT be passed to the secondary (it'd be an unknown model)
    c = await fb.complete("s", "u", model="gemini-grader")
    assert c.text == "groq:None"


def test_get_llm_wraps_with_fallback_by_default():
    s = get_settings()
    s.llm_provider = "gemini"
    s.llm_fallback_provider = "groq"
    assert isinstance(get_llm(s), FallbackLLM)
    # same primary == fallback → no wrapper
    s.llm_fallback_provider = "gemini"
    assert not isinstance(get_llm(s), FallbackLLM)
