"""LLM entity/relation extraction for Graph RAG (ingest-time).

``GraphExtractor.extract(text)`` returns ``{"entities": [...], "relations": [...]}``.
The default implementation uses Claude with a forced tool call so the output is
always valid JSON; tests inject a deterministic fake. Extraction is the expensive
part of building the graph — it runs once per chunk at ingest, never at query time.
"""

from __future__ import annotations

from typing import Any, Protocol

ENTITY_TYPES = ["company", "ticker", "person", "concept", "place", "other"]

_EXTRACT_TOOL = {
    "name": "emit_graph",
    "description": "回報從逐字稿片段抽取出的實體與關係。",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "實體名稱（用最常見的稱呼）"},
                        "type": {"type": "string", "enum": ENTITY_TYPES},
                    },
                    "required": ["name", "type"],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "relation": {"type": "string", "description": "簡短動詞短語，如『供應商』『看多』"},
                        "object": {"type": "string"},
                    },
                    "required": ["subject", "relation", "object"],
                },
            },
        },
        "required": ["entities", "relations"],
    },
}

_SYSTEM = (
    "你是金融逐字稿的知識抽取器。從股癌 podcast 片段中找出重要實體"
    "（公司、股票代號、人物、總經概念、地點）與它們之間的關係。"
    "只抽取片段中明確提到的內容，忽略口語贅詞。務必呼叫 emit_graph 工具回報。"
)


class GraphExtractor(Protocol):
    def extract(self, text: str) -> dict[str, Any]: ...


class ClaudeExtractor:
    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        from anthropic import Anthropic  # lazy

        self._client = Anthropic(api_key=settings.anthropic_api_key or None)
        # Extraction is cheap/structured — Haiku keeps the catalog-wide cost low.
        self.model = getattr(settings, "extract_model", None) or "claude-haiku-4-5"
        self.max_tokens = 2048

    def extract(self, text: str) -> dict[str, Any]:
        import json

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "emit_graph"},
            messages=[{"role": "user", "content": text}],
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "emit_graph":
                # tool inputs may carry odd escaping — round-trip through json to be safe
                data = json.loads(json.dumps(block.input))
                data.setdefault("entities", [])
                data.setdefault("relations", [])
                return data
        return {"entities": [], "relations": []}
