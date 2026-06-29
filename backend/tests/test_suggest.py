"""Query-suggestion / next-step prediction tests (no network)."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag.agentic.llm import Completion
from app.rag.base import RetrievedContext
from app.rag.suggest import QuerySuggester


def _ctx(ep, score, text="輝達財報與台積電供應鏈"):
    return RetrievedContext(text=text, episode_id=ep, ep_number=int(ep[2:]), score=score)


class StubToolbox:
    def __init__(self, hits):
        self._hits = hits

    def execute(self, name, args):
        return "", self._hits


class FakeLLM:
    model = "fake"
    grader_model = "fake"

    def __init__(self, text):
        self._text = text

    async def complete(self, system, user, model=None):
        return Completion(text=self._text, input_tokens=10, output_tokens=10)


def test_is_ambiguous_short_query():
    s = get_settings()
    sg = QuerySuggester(s, toolbox=StubToolbox([]), llm=FakeLLM("{}"))
    amb, _ = sg.is_ambiguous("輝達", [_ctx("EP1", 0.9)])
    assert amb  # too short


def test_is_ambiguous_low_score():
    s = get_settings()
    sg = QuerySuggester(s, toolbox=StubToolbox([]), llm=FakeLLM("{}"))
    amb, _ = sg.is_ambiguous("這個很長的問題但檢索不到", [_ctx("EP1", 0.2)])
    assert amb  # weak retrieval


def test_clear_query_not_ambiguous():
    s = get_settings()
    sg = QuerySuggester(s, toolbox=StubToolbox([]), llm=FakeLLM("{}"))
    amb, _ = sg.is_ambiguous("股癌最近怎麼看美股的修正", [_ctx("EP1", 0.8)])
    assert not amb


@pytest.mark.asyncio
async def test_suggest_uses_llm_when_available():
    s = get_settings()
    s.suggest_use_llm = True
    hits = [_ctx("EP671", 0.3), _ctx("EP659", 0.28)]
    llm = FakeLLM('{"suggestions":["股癌怎麼看輝達財報？","輝達在AI伺服器的角色？","輝達和台積電的關係？"]}')
    res = await QuerySuggester(s, toolbox=StubToolbox(hits), llm=llm).suggest("輝達")
    assert res.ambiguous and len(res.suggestions) == 3
    assert "輝達" in res.suggestions[0].question


@pytest.mark.asyncio
async def test_suggest_falls_back_to_heuristic_on_llm_error():
    class BoomLLM(FakeLLM):
        async def complete(self, system, user, model=None):
            raise RuntimeError("quota exhausted")

    s = get_settings()
    s.suggest_use_llm = True
    hits = [_ctx("EP671", 0.3, "輝達 NVIDIA 財報亮眼"), _ctx("EP659", 0.28, "台積電 先進製程")]
    res = await QuerySuggester(s, toolbox=StubToolbox(hits), llm=BoomLLM("")).suggest("輝達")
    assert res.ambiguous and res.suggestions  # heuristic still produced options
    assert all(sg.episodes for sg in res.suggestions)
