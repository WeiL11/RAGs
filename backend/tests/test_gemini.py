"""Gemini adapter translation tests (no network).

Verifies the Anthropic-block-shape ↔ Gemini Content/Part translation that lets the
agentic tool-loop run on Gemini. Construction uses a dummy key (no API call).
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag.agentic.toolbox import TOOL_SCHEMAS

genai = pytest.importorskip("google.genai")


def _llm():
    from app.rag.agentic.llm import GeminiLLM

    s = get_settings()
    s.gemini_api_key = "dummy-key-no-network"
    s.gemini_model = "gemini-2.5-flash"
    return GeminiLLM(s)


def test_get_llm_selects_provider():
    from app.rag.agentic.llm import AnthropicLLM, GeminiLLM, get_llm

    s = get_settings()
    s.llm_provider = "gemini"
    s.gemini_api_key = "x"
    assert isinstance(get_llm(s), GeminiLLM)
    s.llm_provider = "anthropic"
    assert isinstance(get_llm(s), AnthropicLLM)


def test_tool_schema_translation():
    llm = _llm()
    tools = llm._tool_config(TOOL_SCHEMAS)
    assert tools and len(tools) == 1
    names = [fd.name for fd in tools[0].function_declarations]
    assert "vector_search" in names and "keyword_search" in names


def test_message_translation_roundtrip_shapes():
    llm = _llm()
    messages = [
        {"role": "user", "content": "股癌怎麼看輝達？"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我查一下。"},
                {"type": "tool_use", "id": "gem_0", "name": "vector_search", "input": {"query": "輝達"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "gem_0", "content": "[EP671] 輝達財報亮眼"}
            ],
        },
    ]
    contents = llm._contents(messages)
    assert [c.role for c in contents] == ["user", "model", "tool"]
    # user text
    assert contents[0].parts[0].text == "股癌怎麼看輝達？"
    # assistant carries a function_call mapped from the tool_use block
    fc_parts = [p for p in contents[1].parts if getattr(p, "function_call", None)]
    assert fc_parts and fc_parts[0].function_call.name == "vector_search"
    # tool result becomes a function_response naming the same function (via id->name map)
    fr_parts = [p for p in contents[2].parts if getattr(p, "function_response", None)]
    assert fr_parts and fr_parts[0].function_response.name == "vector_search"
