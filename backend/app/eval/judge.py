"""LLM-as-judge for answer quality (one call per answer — free-tier friendly).

Scores a generated answer against the gold reference + retrieved context on three
0–1 axes: faithfulness (grounded in context, no hallucination), correctness (matches
the reference facts), relevance (actually answers the question). Uses the configured
provider via ``get_llm`` (Gemini on the free deploy).
"""

from __future__ import annotations

import json
import re
from typing import Any

JUDGE_SYSTEM = (
    "You are a strict evaluator of RAG answers for a Traditional-Chinese investing "
    "podcast Q&A. The question, reference and answer are in Chinese; evaluate their "
    "meaning. Output ONLY JSON: "
    '{"faithfulness":<0-1>,"correctness":<0-1>,"relevance":<0-1>}. '
    "faithfulness = every claim is supported by the provided context (no hallucination); "
    "if the answer correctly says it cannot find the info AND the reference says it's "
    "not in the corpus, faithfulness and correctness are 1. "
    "correctness = matches the reference facts. relevance = answers the question."
)


def parse_scores(text: str) -> dict[str, float]:
    s = (text or "").strip()
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1:
        s = s[a : b + 1]
    try:
        d = json.loads(s)
    except Exception:
        return {"faithfulness": 0.0, "correctness": 0.0, "relevance": 0.0}

    def _f(k: str) -> float:
        try:
            return max(0.0, min(1.0, float(d.get(k, 0))))
        except Exception:
            return 0.0

    return {"faithfulness": _f("faithfulness"), "correctness": _f("correctness"), "relevance": _f("relevance")}


async def judge_answer(settings, question, reference: str, answer: str, contexts) -> dict[str, float]:  # type: ignore[no-untyped-def]
    from app.rag.agentic.llm import get_llm

    ctx = "\n".join(f"[{c.episode_id}] {c.text[:300]}" for c in contexts[:5]) or "(none)"
    user = (
        f"Question: {question}\n\nReference answer: {reference or '(none)'}\n\n"
        f"Retrieved context:\n{ctx}\n\nAnswer to grade:\n{answer}"
    )
    comp = await get_llm(settings).complete(JUDGE_SYSTEM, user)
    return parse_scores(comp.text)
