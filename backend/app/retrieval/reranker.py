"""Local cross-encoder reranker (free, no API key).

A bi-encoder (BGE-M3) retrieves fast by comparing independent embeddings, but it
never sees the query and a passage *together*. A cross-encoder does: it scores each
(query, passage) pair jointly, which is slower but markedly more precise. We use it
to re-order a small candidate set — never to search the whole corpus.

The model (``BAAI/bge-reranker-v2-m3``) is multilingual and strong on Traditional
Chinese. It is lazy-loaded so importing this module is cheap and tests can inject a
fake scorer.
"""

from __future__ import annotations

from typing import Protocol

from app.config import Settings, get_settings
from app.rag.base import RetrievedContext


class _Scorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class CrossEncoderReranker:
    """Re-orders retrieved contexts by joint (query, passage) relevance.

    ``model`` can be injected (a fake in tests); otherwise a sentence-transformers
    ``CrossEncoder`` is created lazily on first use.
    """

    def __init__(self, settings: Settings | None = None, *, model: _Scorer | None = None) -> None:
        self._s = settings or get_settings()
        self._model = model

    @property
    def model(self) -> _Scorer:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy, heavy

            self._model = CrossEncoder(self._s.rerank_model)
        return self._model

    def rerank(
        self, query: str, contexts: list[RetrievedContext], top_k: int
    ) -> list[RetrievedContext]:
        """Return the ``top_k`` contexts most relevant to ``query``, best first.

        The cross-encoder score replaces ``score`` (so the UI/eval sort by relevance)
        and ``source`` is tagged ``rerank``. An empty input yields an empty list.
        """
        if not contexts:
            return []
        scores = self.model.predict([(query, c.text) for c in contexts])
        ranked = sorted(zip(contexts, scores), key=lambda cs: float(cs[1]), reverse=True)
        out: list[RetrievedContext] = []
        for c, s in ranked[: max(0, top_k)]:
            c.score = float(s)
            c.source = "rerank"
            out.append(c)
        return out
