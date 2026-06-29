"""Query suggestion / next-step prediction.

When a user's question is too vague to answer well, predict the top-3 questions they
most likely mean — grounded in what's actually in the corpus (via retrieval) — and
offer them as options ("你想問的是？1. 2. 3.").

Works for free: retrieval is local (BGE-M3 + Qdrant), and if no LLM is available it
falls back to a keyword-based heuristic. With an LLM it phrases natural questions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.config import Settings, get_settings
from app.rag.base import RetrievedContext

SUGGEST_SYSTEM = (
    "股癌是一個台灣的『投資理財』podcast，主題多為股票、半導體、科技股與總體經濟。"
    "請特別注意詞彙的投資語境：例如「記憶體」指的是記憶體類股／DRAM 產業，不是人類記憶；"
    "「輝達」「美股」等都是投資標的。"
    "使用者輸入了一個模糊的關鍵字，他想知道的是『股癌在節目中對這個主題的看法或討論』。"
    "請『只』根據下面提供的逐字稿片段內容，提出 3 個具體、扣緊節目實際內容的問題，"
    "句式像「股癌怎麼看…？」「他對…的看法是什麼？」「他提到哪些…？」。"
    "務必使用繁體中文；不要問通識、教科書定義或與投資/節目無關的問題；不要重複。"
    '只輸出 JSON：{"suggestions":["問題一","問題二","問題三"]}'
)


@dataclass
class Suggestion:
    question: str
    episodes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuggestResult:
    ambiguous: bool
    reason: str
    suggestions: list[Suggestion]

    def as_dict(self) -> dict[str, Any]:
        return {"ambiguous": self.ambiguous, "reason": self.reason,
                "suggestions": [s.as_dict() for s in self.suggestions]}


class QuerySuggester:
    def __init__(self, settings: Settings | None = None, *, toolbox=None, llm=None) -> None:
        self._s = settings or get_settings()
        self._toolbox = toolbox
        self._llm = llm

    @property
    def toolbox(self):
        if self._toolbox is None:
            from app.rag.agentic.toolbox import RetrievalToolbox

            self._toolbox = RetrievalToolbox(self._s)
        return self._toolbox

    @property
    def llm(self):
        if self._llm is None:
            from app.rag.agentic.llm import get_llm

            self._llm = get_llm(self._s)
        return self._llm

    def is_ambiguous(self, query: str, hits: list[RetrievedContext]) -> tuple[bool, str]:
        q = (query or "").strip()
        if len(q) < self._s.suggest_min_len:
            return True, "查詢過短"
        top = hits[0].score if hits else 0.0
        if top < self._s.suggest_score_threshold:
            return True, f"最佳檢索分數偏低（{top:.2f}）"
        return False, "查詢明確"

    async def suggest(self, query: str, k: int = 8) -> SuggestResult:
        _, hits = self.toolbox.execute("vector_search", {"query": query, "k": k})
        ambiguous, reason = self.is_ambiguous(query, hits)
        if not ambiguous:
            return SuggestResult(False, reason, [])

        suggestions: list[Suggestion] = []
        if self._s.suggest_use_llm:
            try:
                suggestions = await self._llm_suggest(query, hits)
            except Exception:  # noqa: BLE001 — quota/no-key: degrade to free heuristic
                suggestions = []
        if not suggestions:
            suggestions = self._heuristic_suggest(query, hits)
        return SuggestResult(True, reason, suggestions[:3])

    async def _llm_suggest(self, query: str, hits: list[RetrievedContext]) -> list[Suggestion]:
        ctx = "\n".join(f"[{h.episode_id}] {h.text[:200]}" for h in hits[:6])
        comp = await self.llm.complete(SUGGEST_SYSTEM, f"使用者輸入：{query}\n\n相關片段：\n{ctx}")
        s = comp.text or ""
        a, b = s.find("{"), s.rfind("}")
        data = json.loads(s[a : b + 1]) if a != -1 and b != -1 else {}
        eps = list(dict.fromkeys(h.episode_id for h in hits))
        return [
            Suggestion(question=str(q).strip(), episodes=eps[:1])
            for q in (data.get("suggestions") or [])
            if str(q).strip()
        ]

    def _heuristic_suggest(self, query: str, hits: list[RetrievedContext]) -> list[Suggestion]:
        """Free fallback: per top distinct episode, the most frequent meaningful (CJK,
        non-filler) terms become a topic hint. Rough vs the LLM, but $0 and offline."""
        import re
        from collections import Counter, defaultdict

        import jieba

        skip = set(jieba.lcut(query)) | _FILLER
        by_ep: dict[str, list[RetrievedContext]] = defaultdict(list)
        order: list[str] = []
        for h in hits:
            if h.episode_id not in by_ep:
                order.append(h.episode_id)
            by_ep[h.episode_id].append(h)

        out: list[Suggestion] = []
        for ep in order[:3]:
            cnt: Counter[str] = Counter()
            for h in by_ep[ep]:
                for t in jieba.lcut(h.text):
                    if len(t) >= 2 and t not in skip and re.fullmatch(r"[一-鿿]+", t):
                        cnt[t] += 1
            terms = [t for t, _ in cnt.most_common(2)]
            if not terms:
                continue
            out.append(Suggestion(
                question=f"你想了解「{'、'.join(terms)}」相關的內容嗎？", episodes=[ep]))
        return out


_FILLER = {
    "覺得", "可能", "東西", "然後", "就是", "這個", "那個", "他們", "我們", "自己", "的話",
    "一個", "什麼", "沒有", "因為", "所以", "可以", "大家", "時候", "現在", "其實", "比較",
    "這樣", "這邊", "知道", "應該", "不會", "如果", "或是", "已經", "還是", "真的", "一些",
    "起來", "出來", "今天", "問題", "看到", "講說",
}
