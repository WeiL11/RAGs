"""Minimal Gradio chat for the Gooaye RAG demo (Hugging Face Spaces).

A single dialog: ask a question, get a cited answer. Uses the free Gemini provider
and the prebuilt local index shipped under ./data. Env is set before importing the
backend so the shipped index + Gemini provider are picked up.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LLM_PROVIDER", "groq")
# Keep the query-suggestion feature free on the Space (heuristic, no LLM call).
os.environ.setdefault("SUGGEST_USE_LLM", "false")
os.environ.setdefault("EMBED_PROVIDER", "local")
os.environ.setdefault("QDRANT_MODE", "local")
os.environ.setdefault("QDRANT_PATH", os.path.join(HERE, "data", "qdrant_local"))
os.environ.setdefault("TRANSCRIPTS_DIR", os.path.join(HERE, "data", "transcripts"))
os.environ.setdefault("GRAPH_PATH", os.path.join(HERE, "data", "graph.json"))
# Graph RAG only works if a prebuilt graph.json was shipped; enable it conditionally.
_strats = "corrective,agentic"
if os.path.exists(os.environ["GRAPH_PATH"]):
    _strats += ",graph"
os.environ.setdefault("ENABLED_STRATEGIES", _strats)
# GEMINI_API_KEY comes from the Space secret.

import gradio as gr  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.registry import build_default_registry  # noqa: E402
from app.rag.suggest import QuerySuggester  # noqa: E402

settings = get_settings()
registry = build_default_registry(settings)
suggester = QuerySuggester(settings)
STRATEGIES = registry.names()

try:
    from app.ingestion.status import corpus_range  # noqa: E402

    CORPUS = corpus_range(settings).human
except Exception:  # noqa: BLE001
    CORPUS = ""


def _provider_key(s) -> str:
    return {"groq": s.groq_api_key, "gemini": s.gemini_api_key,
            "anthropic": s.anthropic_api_key}.get(s.llm_provider, "")


def _ts(s: float | None) -> str:
    s = s or 0
    return f"{int(s // 60)}:{int(s % 60):02d}"


async def respond(message: str, history, strategy: str):
    if not (message or "").strip():
        yield "請輸入問題。"
        return

    # Next-step prediction: if the question is too vague, offer the top-3 likely questions
    # instead of answering. Retrieval-grounded; free (heuristic) on the Space.
    try:
        sug = await suggester.suggest(message)
    except Exception:  # noqa: BLE001
        sug = None
    if sug and sug.ambiguous and sug.suggestions:
        opts = "\n".join(f"{i}. {s.question}" for i, s in enumerate(sug.suggestions, 1))
        yield f"🤔 你的問題有點籠統（{sug.reason}），你想問的是？\n\n{opts}\n\n（輸入更完整的問題，我就會直接回答）"
        return

    if not _provider_key(settings):
        yield f"⚠️ 尚未設定 {settings.llm_provider.upper()}_API_KEY（請在 Space 的 Settings → Secrets 加入）。"
        return
    try:
        strat = registry.get(strategy)
    except KeyError:
        yield f"未知策略：{strategy}"
        return

    answer, sources = "", []
    try:
        async for ev in strat.answer(message):
            if ev.type == "token":
                answer += ev.delta
                yield answer
            elif ev.type == "contexts":
                sources = ev.contexts
            elif ev.type == "error":
                yield f"⚠️ {ev.delta}"
                return
    except Exception as exc:  # noqa: BLE001
        yield f"⚠️ {type(exc).__name__}: {exc}"
        return

    if sources:
        lines = "\n".join(f"- {c.episode_id} @ {_ts(c.start_s)}" for c in sources[:5])
        answer += f"\n\n---\n**來源 Sources**\n{lines}"
    yield answer or "（沒有產生答案）"


demo = gr.ChatInterface(
    fn=respond,
    title="股癌 Gooaye — Podcast RAG",
    description=(f"📚 涵蓋最近集數：{CORPUS}。\n" if CORPUS else "")
    + "問股癌 podcast 的內容，AI 依逐字稿回答並標註來源。內容為節目個人觀點，非投資建議。",
    additional_inputs=[
        gr.Dropdown(choices=STRATEGIES, value=STRATEGIES[0], label="RAG strategy")
    ],
    examples=[
        ["股癌最近怎麼看美股？", STRATEGIES[0]],
        ["他對記憶體類股的看法？", STRATEGIES[0]],
        ["他對比特幣的看法？", STRATEGIES[0]],
    ],
)

if __name__ == "__main__":
    demo.launch()
