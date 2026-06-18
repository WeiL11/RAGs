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


def _parse_extraction(text: str) -> dict[str, Any]:
    """Robustly pull {entities, relations} out of an LLM JSON reply."""
    import json
    import re

    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1:
        s = s[a : b + 1]
    try:
        data = json.loads(s)
    except Exception:
        return {"entities": [], "relations": []}
    return {
        "entities": data.get("entities", []) or [],
        "relations": data.get("relations", []) or [],
    }


_GEMINI_SYSTEM = (
    _SYSTEM
    + "\n只輸出 JSON，格式："
    + '{"entities":[{"name":"...","type":"company|ticker|person|concept|place|other"}],'
    + '"relations":[{"subject":"...","relation":"...","object":"..."}]}'
)


class GeminiExtractor:
    """Free entity/relation extraction via Gemini JSON mode."""

    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        from google import genai  # lazy
        from google.genai import types  # lazy

        self._types = types
        self._client = genai.Client(api_key=settings.gemini_api_key or None)
        self.model = settings.gemini_model

    def extract(self, text: str) -> dict[str, Any]:
        import time

        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=_GEMINI_SYSTEM,
            response_mime_type="application/json",
            max_output_tokens=2048,
        )
        last_err: Exception | None = None
        for attempt in range(3):  # tolerate free-tier rate limits
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=text, config=config
                )
                return _parse_extraction(resp.text)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Gemini extraction failed after retries: {last_err}")


def get_extractor(settings):  # type: ignore[no-untyped-def]
    """Pick the extractor matching the configured LLM provider."""
    if settings.llm_provider == "gemini":
        return GeminiExtractor(settings)
    return ClaudeExtractor(settings)
