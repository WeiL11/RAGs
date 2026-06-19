"""Groq adapter translation tests (no network) — Anthropic block shape → OpenAI chat."""

from __future__ import annotations

import json

import pytest

from app.config import get_settings

pytest.importorskip("groq")


def _llm():
    from app.rag.agentic.llm import GroqLLM

    s = get_settings()
    s.groq_api_key = "dummy"
    s.groq_model = "llama-3.3-70b-versatile"
    return GroqLLM(s)


def test_get_llm_and_extractor_select_groq():
    from app.config import get_settings
    from app.rag.agentic.llm import GroqLLM, get_llm
    from app.rag.graph.extract import GroqExtractor, get_extractor

    s = get_settings()
    s.llm_provider = "groq"
    s.groq_api_key = "x"
    assert isinstance(get_llm(s), GroqLLM)
    assert isinstance(get_extractor(s), GroqExtractor)


def test_tools_to_openai_shape():
    from app.rag.agentic.toolbox import TOOL_SCHEMAS

    tools = _llm()._tools(TOOL_SCHEMAS)
    assert tools[0]["type"] == "function"
    names = [t["function"]["name"] for t in tools]
    assert "vector_search" in names


def test_messages_to_openai_shape():
    llm = _llm()
    messages = [
        {"role": "user", "content": "股癌怎麼看輝達？"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "查一下"},
            {"type": "tool_use", "id": "c1", "name": "vector_search", "input": {"query": "輝達"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "[EP671] 輝達財報亮眼"},
        ]},
    ]
    out = llm._messages("SYS", messages)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "股癌怎麼看輝達？"}
    # assistant turn carries an OpenAI tool_call
    asst = out[2]
    assert asst["role"] == "assistant" and asst["tool_calls"][0]["function"]["name"] == "vector_search"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"query": "輝達"}
    # tool result becomes a 'tool' message keyed by tool_call_id
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "c1"
