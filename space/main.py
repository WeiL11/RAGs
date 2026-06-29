"""Minimal Gradio chat for the Gooaye RAG demo (Hugging Face Spaces).

A single dialog: ask a question, get a cited answer. Uses the free Gemini provider
and the prebuilt local index shipped under ./data. Env is set before importing the
backend so the shipped index + Gemini provider are picked up.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("LLM_PROVIDER", "gemini")  # primary; groq is the auto-fallback ("option B")
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

import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import gradio as gr  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.ingestion.models import TranscriptDoc  # noqa: E402
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


# Best strategy by a small eval: corrective (faithfulness 0.93 / relevance 0.97).
# agentic is unreliable on Groq's tool-calling, so the demo is fixed to corrective.
BEST_STRATEGY = "corrective" if "corrective" in STRATEGIES else STRATEGIES[0]

# Fixed disclaimer appended to every answer (not LLM-generated, so it's consistent).
DISCLAIMER = "⚠️ 以上為節目個人觀點整理，非投資建議。"


async def respond(message: str, history):
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
    strat = registry.get(BEST_STRATEGY)

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

    extras = ""
    if sources:
        lines = "\n".join(f"- {c.episode_id} @ {_ts(c.start_s)}" for c in sources[:5])
        extras = f"\n\n---\n**來源 Sources**\n{lines}"
    yield (answer or "（沒有產生答案）") + extras + f"\n\n_{DISCLAIMER}_"


_DESC = (
    (f"📚 涵蓋最近集數：{CORPUS}。\n\n" if CORPUS else "")
    + "🏆 目前最佳方法：**Corrective RAG**（小樣本評估：忠實度 0.93、相關性 0.97；agentic 在 Groq 上的 function-calling 不穩）。\n\n"
    + "問股癌 podcast 的內容，AI 依逐字稿回答並標註來源（時間戳）。內容為節目個人觀點，非投資建議。"
)

# Transcript viewer (lazy): list episodes, load one only when selected (avoids slow startup).
_EP_FILES = {Path(p).stem: p for p in glob.glob(os.path.join(HERE, "data", "transcripts", "*.json"))}
_EP_LIST = sorted(_EP_FILES, key=lambda e: int(e[2:]) if e[2:].isdigit() else 0, reverse=True)


def load_transcript(ep: str) -> str:
    if not ep or ep not in _EP_FILES:
        return ""
    doc = TranscriptDoc.load(Path(_EP_FILES[ep]))
    blocks, start, buf = [], None, []
    for seg in doc.segments:  # group into ~30s lines so it renders fast + reads cleanly
        if start is None:
            start = seg.start
        buf.append(seg.text)
        if seg.end - start >= 30:
            blocks.append(f"**`{_ts(start)}`**　{''.join(buf)}")
            start, buf = None, []
    if buf and start is not None:
        blocks.append(f"**`{_ts(start)}`**　{''.join(buf)}")
    return f"#### {ep} · {doc.publish_date} · {len(doc.segments)} 段\n\n" + "\n\n".join(blocks)


async def _chat(message, history):
    if not (message or "").strip():
        yield history or [], ""
        return
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield history, ""
    async for partial in respond(message, history[:-1]):
        history[-1]["content"] = partial
        yield history, ""


with gr.Blocks(title="股癌 Gooaye — Podcast RAG", fill_height=True) as demo:
    gr.Markdown(f"# 股癌 Gooaye — Podcast RAG\n\n{_DESC}")
    chatbot = gr.Chatbot(height=400, show_label=False)
    box = gr.Textbox(placeholder="例如：股癌最近怎麼看美股？（打模糊的詞會給你建議）", show_label=False, autofocus=True)
    gr.Examples(["股癌最近怎麼看美股？", "他對記憶體類股的看法？", "他對比特幣的看法？"], inputs=box)
    box.submit(_chat, [box, chatbot], [chatbot, box])

    with gr.Accordion("📜 逐字稿檢視（核對時間軸 — 選了集數才載入，避免拖慢）", open=False):
        ep_dd = gr.Dropdown(_EP_LIST, label="選擇集數", value=None)
        ep_view = gr.Markdown()
        ep_dd.change(load_transcript, ep_dd, ep_view)


if __name__ == "__main__":
    demo.launch()
