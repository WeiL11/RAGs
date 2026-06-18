"""Graph RAG: store mechanics, builder (fake extractor), and strategy (fake LLM)."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from app.config import get_settings
from app.ingestion.build_graph import build_graph
from app.ingestion.models import Segment, TranscriptDoc
from app.rag.agentic.llm import TurnResult
from app.rag.graph.strategy import GraphRAGStrategy
from app.retrieval.graph_store import GraphStore


class SingleTurnLLM:
    model = "claude-opus-4-8"

    def __init__(self, text: str) -> None:
        self._text = text

    async def stream_turn(self, system, messages, tools) -> AsyncIterator[tuple[str, Any]]:
        # capture the prompt so the test can assert grounding
        self.last_user = messages[-1]["content"]
        for ch in self._text:
            yield ("delta", ch)
        yield ("final", TurnResult(self._text, [], [], "end_turn", 120, 60))


class FakeExtractor:
    """Deterministic extraction keyed by substrings, for build_graph tests."""

    def extract(self, text: str) -> dict[str, Any]:
        ents, rels = [], []
        if "輝達" in text:
            ents.append({"name": "輝達", "type": "company"})
        if "台積電" in text:
            ents.append({"name": "台積電", "type": "company"})
        if "輝達" in text and "台積電" in text:
            rels.append({"subject": "輝達", "relation": "供應商", "object": "台積電"})
        return {"entities": ents, "relations": rels}


def test_graph_store_link_traverse_provenance():
    g = GraphStore(":memory:")  # path unused until save()
    g.add_entity("輝達", "company", {"text": "輝達財報亮眼", "episode_id": "EP512",
                                     "ep_number": 512, "publish_date": "2024-12-07",
                                     "start_s": 10.0, "end_s": 40.0})
    g.add_entity("台積電", "company", {"text": "台積電先進製程領先", "episode_id": "EP512",
                                       "ep_number": 512, "publish_date": "2024-12-07",
                                       "start_s": 40.0, "end_s": 70.0})
    g.add_relation("輝達", "供應商", "台積電", "EP512", "輝達與台積電合作")

    seeds = g.link_entities("股癌怎麼看輝達", ["輝達"])
    assert "輝達" in [s for s in seeds]
    nbrs = g.neighbors("輝達", hops=1)
    assert "台積電" in nbrs  # reached via the relation edge
    triples = g.relation_triples({"輝達", "台積電"})
    assert any("供應商" in t for t in triples)
    ctxs = g.mentions_as_contexts({"輝達", "台積電"})
    assert {c.episode_id for c in ctxs} == {"EP512"}
    assert all(c.source == "graph" for c in ctxs)


def test_build_graph_from_transcripts(tmp_path):
    s = get_settings()
    s.transcripts_dir = str(tmp_path / "t")
    s.graph_path = str(tmp_path / "graph.json")
    from pathlib import Path

    Path(s.transcripts_dir).mkdir(parents=True)
    TranscriptDoc(
        episode_id="EP512", ep_number=512, title="t", publish_date="2024-12-07",
        duration_s=3000, asr_provider="mlx", asr_model="large-v3",
        segments=[Segment(0, 30, "今天聊聊輝達與台積電的合作關係非常緊密。" * 10)],
    ).save(Path(s.transcripts_dir) / "EP512.json")

    store = build_graph(s, extractor=FakeExtractor())
    assert store.num_nodes >= 2
    assert store.num_edges >= 1
    # persisted graph reloads
    reloaded = GraphStore(s.graph_path).load()
    assert reloaded.num_nodes == store.num_nodes


@pytest.mark.asyncio
async def test_graph_strategy_grounds_and_traces():
    g = GraphStore(":memory:")
    g.add_entity("輝達", "company", {"text": "輝達財報亮眼，資料中心需求強。", "episode_id": "EP512",
                                     "ep_number": 512, "publish_date": "2024-12-07",
                                     "start_s": 10.0, "end_s": 40.0})
    llm = SingleTurnLLM("根據逐字稿，股癌看好輝達（EP512）。這不是投資建議。")
    strat = GraphRAGStrategy(get_settings(), graph=g, llm=llm)

    events = [e async for e in strat.answer("股癌怎麼看輝達？")]
    assert events[0].type == "contexts"
    assert events[-1].type == "done"
    answer = "".join(e.delta for e in events if e.type == "token")
    assert "輝達" in answer
    # prompt was grounded in the retrieved chunk
    assert "輝達財報亮眼" in llm.last_user
    assert events[-1].trace["n_contexts"] == 1
    assert events[-1].trace["cost_usd"] > 0
