"""M4 eval-harness logic tests (no LLM / no network)."""

from __future__ import annotations

from app.eval.judge import parse_scores
from app.eval.run_eval import aggregate, retrieval_recall
from app.rag.base import RetrievedContext


def _ctx(ep: str):
    return RetrievedContext(text="x", episode_id=ep, ep_number=int(ep[2:]), score=1.0)


def test_parse_scores_clamps_and_defaults():
    assert parse_scores('{"faithfulness":0.9,"correctness":1,"relevance":0.5}') == {
        "faithfulness": 0.9, "correctness": 1.0, "relevance": 0.5,
    }
    # clamp out-of-range + tolerate noise around the JSON
    assert parse_scores('ok: {"faithfulness":2,"correctness":-1,"relevance":0.3} done')["faithfulness"] == 1.0
    assert parse_scores("not json") == {"faithfulness": 0.0, "correctness": 0.0, "relevance": 0.0}


def test_retrieval_recall():
    assert retrieval_recall([_ctx("EP1"), _ctx("EP2")], ["EP1", "EP2"]) == 1.0
    assert retrieval_recall([_ctx("EP1"), _ctx("EP9")], ["EP1", "EP2"]) == 0.5
    assert retrieval_recall([_ctx("EP9")], ["EP1"]) == 0.0
    assert retrieval_recall([_ctx("EP9")], []) is None  # negative question


def test_aggregate_means_and_recall_ignores_none():
    rows = [
        {"strategy": "graph", "faithfulness": 1.0, "correctness": 0.8, "relevance": 1.0,
         "retrieval_recall": 1.0, "latency_ms": 100, "error": None},
        {"strategy": "graph", "faithfulness": 0.0, "correctness": 0.2, "relevance": 0.0,
         "retrieval_recall": None, "latency_ms": 300, "error": "boom"},
    ]
    card = aggregate(rows)["graph"]
    assert card["faithfulness"] == 0.5
    assert card["retrieval_recall"] == 1.0  # the None is ignored
    assert card["latency_ms"] == 200
    assert card["errors"] == 1.0
