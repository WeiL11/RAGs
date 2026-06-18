"""A thin streaming-LLM seam so the agentic loop is testable without the network.

The real implementation (``AnthropicLLM``) calls the Anthropic SDK's streaming
helper; tests inject a scripted fake. Both yield ``("delta", text)`` events as the
answer streams, then a final ``("final", TurnResult)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def stream_turn(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[tuple[str, Any]]: ...

    async def complete(self, system: str, user: str, model: str | None = None) -> Completion:
        """Non-streaming single completion (used by CRAG grading/rewriting)."""
        ...


class AnthropicLLM:
    """Default LLM: Anthropic SDK streaming with adaptive thinking (Opus 4.8)."""

    def __init__(self, settings) -> None:  # type: ignore[no-untyped-def]
        from anthropic import AsyncAnthropic  # lazy

        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key or None)
        self.model = settings.answer_model
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
