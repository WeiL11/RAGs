"""Streaming-LLM seam, provider-pluggable.

Two real adapters — ``AnthropicLLM`` (Claude) and ``GeminiLLM`` (free Gemini) — plus
``get_llm(settings)`` to pick one. Tests inject a scripted fake. All expose:
  - ``stream_turn(system, messages, tools)`` → async-yields ("delta", text) then
    ("final", TurnResult). ``messages``/``tools`` use the Anthropic block shape as the
    INTERNAL format; GeminiLLM translates it to/from Gemini function-calling.
  - ``complete(system, user, model)`` → one-shot ``Completion`` (used by CRAG).
  - ``model`` and ``grader_model`` attributes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class TurnResult:
    text: str
    tool_uses: list[ToolUse]
    raw_content: Any  # assistant content blocks, appended verbatim to messages
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    model: str
    grader_model: str

    def stream_turn(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[tuple[str, Any]]: ...

    async def complete(self, system: str, user: str, model: str | None = None) -> Completion: ...


def get_llm(settings) -> LLM:  # type: ignore[no-untyped-def]
    """Pick the adapter from settings.llm_provider ('anthropic' | 'gemini')."""
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings)
    return AnthropicLLM(settings)


class AnthropicLLM:
    """Claude via the Anthropic SDK, streaming with adaptive thinking."""

    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        from anthropic import AsyncAnthropic  # lazy

        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key or None)
        self.model = settings.answer_model
        self.grader_model = settings.grader_model
        self.max_tokens = settings.max_tokens

    async def stream_turn(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[tuple[str, Any]]:
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive"},
        ) as stream:
            async for text in stream.text_stream:
                yield ("delta", text)
            final = await stream.get_final_message()

        tool_uses = [
            ToolUse(id=b.id, name=b.name, input=dict(b.input))
            for b in final.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in final.content if b.type == "text")
        yield (
            "final",
            TurnResult(
                text=text,
                tool_uses=tool_uses,
                raw_content=final.content,
                stop_reason=final.stop_reason or "end_turn",
                input_tokens=getattr(final.usage, "input_tokens", 0) or 0,
                output_tokens=getattr(final.usage, "output_tokens", 0) or 0,
            ),
        )

    async def complete(self, system: str, user: str, model: str | None = None) -> Completion:
        msg = await self._client.messages.create(
            model=model or self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Completion(
            text=text,
            input_tokens=getattr(msg.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(msg.usage, "output_tokens", 0) or 0,
        )


# --- block-shape helpers (work on dicts or SDK objects) -----------------------


def _btype(b: Any) -> str | None:
    return b.get("type") if isinstance(b, dict) else getattr(b, "type", None)


def _bget(b: Any, key: str, default: Any = None) -> Any:
    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)


class GeminiLLM:
    """Gemini via google-genai, with MANUAL function calling.

    Translates the internal Anthropic block shape ↔ Gemini ``Content``/``Part`` so the
    agentic tool-loop (and graph/corrective) run unchanged. Manual function calling
    avoids the known auto-function-calling streaming bug.
    """

    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        from google import genai  # lazy
        from google.genai import types  # lazy

        self._types = types
        self._client = genai.Client(api_key=settings.gemini_api_key or None)
        self.model = settings.gemini_model
        self.grader_model = settings.gemini_model
        self.max_tokens = settings.max_tokens

    def _tool_config(self, tools: list[dict[str, Any]]):
        if not tools:
            return None
        t = self._types
        decls = [
            t.FunctionDeclaration(
                name=x["name"],
                description=x.get("description", ""),
                parameters_json_schema=x["input_schema"],
            )
            for x in tools
        ]
        return [t.Tool(function_declarations=decls)]

    def _contents(self, messages: list[dict[str, Any]]):
        """Anthropic-shape messages → list[types.Content]."""
        t = self._types
        # map tool_use id → name so tool_result blocks can name their functionResponse
        id2name: dict[str, str] = {}
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                for b in c:
                    if _btype(b) == "tool_use":
                        id2name[_bget(b, "id")] = _bget(b, "name")

        contents = []
        for m in messages:
            role, c = m["role"], m.get("content")
            if isinstance(c, str):
                contents.append(
                    t.Content(role="user" if role == "user" else "model", parts=[t.Part(text=c)])
                )
                continue
            blocks = c or []
            if any(_btype(b) == "tool_result" for b in blocks):
                parts = [
                    t.Part.from_function_response(
                        name=id2name.get(_bget(b, "tool_use_id"), "tool"),
                        response={"result": _bget(b, "content", "")},
                    )
                    for b in blocks
                    if _btype(b) == "tool_result"
                ]
                contents.append(t.Content(role="tool", parts=parts))
            else:
                parts = []
                for b in blocks:
                    if _btype(b) == "text" and _bget(b, "text"):
                        parts.append(t.Part(text=_bget(b, "text")))
                    elif _btype(b) == "tool_use":
                        parts.append(
                            t.Part(
                                function_call=t.FunctionCall(
                                    name=_bget(b, "name"), args=dict(_bget(b, "input", {}))
                                )
                            )
                        )
                contents.append(t.Content(role="model", parts=parts or [t.Part(text="")]))
        return contents

    async def stream_turn(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[tuple[str, Any]]:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system or None,
            tools=self._tool_config(tools),
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=self.max_tokens,
        )
        stream = self._client.aio.models.generate_content_stream(
            model=self.model, contents=self._contents(messages), config=config
        )
        if inspect.isawaitable(stream):
            stream = await stream

        text_parts: list[str] = []
        fcalls: list[Any] = []
        usage = None
        async for chunk in stream:
            if getattr(chunk, "text", None):
                yield ("delta", chunk.text)
                text_parts.append(chunk.text)
            cands = getattr(chunk, "candidates", None) or []
            if cands and cands[0].content and cands[0].content.parts:
                for p in cands[0].content.parts:
                    if getattr(p, "function_call", None):
                        fcalls.append(p.function_call)
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata

        text = "".join(text_parts)
        tool_uses = [
            ToolUse(id=f"gem_{i}", name=fc.name, input=dict(fc.args or {}))
            for i, fc in enumerate(fcalls)
        ]
        raw: list[dict[str, Any]] = []
        if text:
            raw.append({"type": "text", "text": text})
        raw += [{"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input} for tu in tool_uses]
        yield (
            "final",
            TurnResult(
                text=text,
                tool_uses=tool_uses,
                raw_content=raw,
                stop_reason="tool_use" if tool_uses else "end_turn",
                input_tokens=(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
                output_tokens=(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
            ),
        )

    async def complete(self, system: str, user: str, model: str | None = None) -> Completion:
        t = self._types
        resp = await self._client.aio.models.generate_content(
            model=model or self.model,
            contents=user,
            config=t.GenerateContentConfig(system_instruction=system or None, max_output_tokens=1024),
        )
        u = getattr(resp, "usage_metadata", None)
        return Completion(
            text=resp.text or "",
            input_tokens=(getattr(u, "prompt_token_count", 0) or 0) if u else 0,
            output_tokens=(getattr(u, "candidates_token_count", 0) or 0) if u else 0,
        )
