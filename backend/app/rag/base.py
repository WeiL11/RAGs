"""The stable RAG contract.

Everything in the app shell (FastAPI routes, frontend) depends ONLY on the types
and the ``BaseRAGStrategy`` interface in this module. RAG strategies are swapped
freely behind this contract — adding or iterating a strategy must never require a
change to ``main.py`` or the frontend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator


@dataclass
class Message:
    """A single chat turn."""

    role: str  # "user" | "assistant"
    content: str


@dataclass
class RetrievedContext:
    """One retrieved chunk, carrying enough metadata to cite + filter by time.

    ``start_s`` / ``end_s`` are second offsets into the episode audio so the UI can
    deep-link to the exact moment. ``source`` records which retriever produced it
    (useful when a strategy fuses vector + graph + keyword hits).
    """

    text: str
    episode_id: str
    ep_number: int
    publish_date: str | None = None  # ISO-8601 date, e.g. "2024-12-07"
    start_s: float | None = None
    end_s: float | None = None
    score: float = 0.0
    source: str = "unknown"  # "vector" | "graph" | "keyword" | "stub"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StreamEvent:
    """A single server-sent event in an answer stream.

    ``type``:
      - "contexts": retrieval finished; ``contexts`` is populated.
      - "token":    an incremental piece of the answer in ``delta``.
      - "done":     stream complete; ``trace`` holds latency/tokens/cost/tool-calls.
      - "error":    something failed; ``delta`` holds a human-readable message.
    """

    type: str
    delta: str = ""
    contexts: list[RetrievedContext] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.delta:
            d["delta"] = self.delta
        if self.contexts:
            d["contexts"] = [c.as_dict() for c in self.contexts]
        if self.trace:
            d["trace"] = self.trace
        return d


class BaseRAGStrategy(ABC):
    """Implement this once per RAG approach (agentic, graph, ...).

    Subclasses set ``name`` (the stable id used by the API/frontend) and
    ``description`` (shown in the strategy dropdown), then implement ``answer``.
    ``retrieve`` is provided for the eval harness (context precision/recall) and
    defaults to draining an ``answer`` stream's "contexts" event.
    """

    name: str = "base"
    description: str = ""

    @abstractmethod
    async def answer(
        self,
        query: str,
        history: list[Message] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the answer for ``query``. Must yield a final "done" event."""
        raise NotImplementedError
        yield  # pragma: no cover  (makes this an async generator for typing)

    async def retrieve(
        self,
        query: str,
        k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedContext]:
        """Return the contexts this strategy would use for ``query``.

        Default implementation runs ``answer`` and captures its "contexts" event,
        so eval works for any strategy out of the box. Strategies with a cheaper
        retrieval-only path should override this.
        """
        contexts: list[RetrievedContext] = []
        async for event in self.answer(query, history=None, filters=filters):
            if event.type == "contexts":
                contexts = event.contexts
                break
        return contexts[:k]
