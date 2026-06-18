"""Config-driven registry of RAG strategies.

The app shell asks the registry which strategies exist and looks them up by name.
Wiring a new strategy = implement ``BaseRAGStrategy`` + add one line to
``build_default_registry``. No route or frontend change required.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.rag.base import BaseRAGStrategy


class StrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, BaseRAGStrategy] = {}

    def register(self, strategy: BaseRAGStrategy) -> None:
        if not strategy.name:
            raise ValueError("strategy.name must be set")
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> BaseRAGStrategy:
        if name not in self._strategies:
            raise KeyError(name)
        return self._strategies[name]

    def names(self) -> list[str]:
        return list(self._strategies)

    def describe(self) -> list[dict[str, str]]:
        """Payload for ``GET /strategies`` — drives the frontend dropdown."""
        return [
            {"name": s.name, "description": s.description}
            for s in self._strategies.values()
        ]


def build_default_registry(settings: Settings | None = None) -> StrategyRegistry:
    """Construct the registry based on settings.

    Strategies are registered in the order listed in ``settings.enabled_strategies``;
    the first one becomes the UI default. Heavy strategies (agentic, graph) are
    imported lazily so M0 runs without their deps installed.
    """
    settings = settings or get_settings()
    registry = StrategyRegistry()

    for name in settings.enabled_strategies:
        if name == "echo":
            from app.rag.echo.strategy import EchoStrategy

            registry.register(EchoStrategy())
        elif name == "agentic":
            from app.rag.agentic.strategy import AgenticRAGStrategy

            registry.register(AgenticRAGStrategy(settings))
        elif name == "graph":
            from app.rag.graph.strategy import GraphRAGStrategy

            registry.register(GraphRAGStrategy(settings))
        elif name == "corrective":
            from app.rag.corrective.strategy import CorrectiveRAGStrategy

            registry.register(CorrectiveRAGStrategy(settings))
        else:
            raise ValueError(f"Unknown strategy in enabled_strategies: {name!r}")

    if not registry.names():
        raise RuntimeError("No strategies enabled — check ENABLED_STRATEGIES config")
    return registry
