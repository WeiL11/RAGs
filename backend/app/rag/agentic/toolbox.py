"""Retrieval tools exposed to the agentic loop.

Each tool returns ``(result_text, contexts)``: ``result_text`` is the string fed
back to Claude as the tool_result; ``contexts`` are the ``RetrievedContext`` objects
surfaced to the UI / eval. Tools are lazy about their backends so the module imports
without qdrant/voyage installed, and tests can inject stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.ingestion.models import TranscriptDoc
from app.rag.base import RetrievedContext

# JSON-schema tool definitions sent to the Messages API.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "vector_search",
        "description": (
            "語意檢索股癌 podcast 逐字稿，找出與查詢最相關的片段。"
            "適合概念、主題、觀點類查詢。可用日期或集數範圍過濾。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "檢索查詢（繁體中文）"},
                "k": {"type": "integer", "description": "回傳片段數，預設 8", "default": 8},
                "date_from": {"type": "string", "description": "起始日期 YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                "ep_min": {"type": "integer", "description": "最小集數"},
                "ep_max": {"type": "integer", "description": "最大集數"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "keyword_search",
        "description": (
            "關鍵字 (BM25) 檢索逐字稿，擅長精準詞彙：股票代號、公司名、人名、數字。"
            "當語意檢索找不到特定名詞時使用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_episode_metadata",
        "description": "取得某一集的標題、發布日期、長度與片段數。",
        "input_schema": {
            "type": "object",
            "properties": {"ep_number": {"type": "integer"}},
            "required": ["ep_number"],
        },
    },
    {
        "name": "expand_context",
        "description": (
            "取得某一集在指定時間區間前後的完整逐字稿，用來擴充某個片段的上下文。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "例如 EP512"},
                "start_s": {"type": "number"},
                "end_s": {"type": "number"},
                "window_s": {"type": "number", "description": "前後擴充秒數，預設 60", "default": 60},
            },
            "required": ["episode_id", "start_s", "end_s"],
        },
    },
]


class RetrievalToolbox:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder=None,
        store=None,
        keyword_index=None,
    ) -> None:
        self._s = settings or get_settings()
        self._embedder = embedder
        self._store = store
        self._kw = keyword_index
        self._index_verified = False

    # --- lazy backends ---------------------------------------------------------

    @property
    def embedder(self):
        if self._embedder is None:
            from app.retrieval.embedder import get_embedder

            self._embedder = get_embedder(self._s)
        return self._embedder

    @property
    def store(self):
        if self._store is None:
            from app.retrieval.vector_store import VectorStore

            self._store = VectorStore(self._s)
        return self._store

    @property
    def keyword_index(self):
        if self._kw is None:
            from app.retrieval.keyword import KeywordIndex

            self._kw = KeywordIndex(self._s).build()
        return self._kw

    # --- dispatch --------------------------------------------------------------

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, list[RetrievedContext]]:
        try:
            if name == "vector_search":
                return self._vector_search(args)
            if name == "keyword_search":
                return self._keyword_search(args)
            if name == "get_episode_metadata":
                return self._episode_metadata(args), []
            if name == "expand_context":
                return self._expand_context(args)
        except Exception as exc:  # surface as a tool error, don't crash the loop
            return f"工具錯誤: {type(exc).__name__}: {exc}", []
        return f"未知工具: {name}", []

    # --- tools -----------------------------------------------------------------

    def _verify_index(self) -> None:
        """Guard: ensure the live vector index matches the current embedder/chunk config."""
        if self._index_verified:
            return
        from app.retrieval.vector_store import build_manifest

        self.store.assert_manifest(build_manifest(self._s, self.embedder))
        self._index_verified = True

    def _vector_search(self, args: dict[str, Any]) -> tuple[str, list[RetrievedContext]]:
        self._verify_index()
        k = int(args.get("k", 8))
        filters = {
            key: args[key]
            for key in ("date_from", "date_to", "ep_min", "ep_max")
            if args.get(key) is not None
        }
        vec = self.embedder.embed_query(args["query"])
        hits = self.store.search(vec, k=k, filters=filters or None)
        return _format_hits(hits), hits

    def _keyword_search(self, args: dict[str, Any]) -> tuple[str, list[RetrievedContext]]:
        hits = self.keyword_index.search(args["query"], k=int(args.get("k", 8)))
        return _format_hits(hits), hits

    def _episode_metadata(self, args: dict[str, Any]) -> str:
        doc = self._load(f"EP{int(args['ep_number'])}")
        if not doc:
            return f"找不到 EP{args['ep_number']} 的逐字稿。"
        return (
            f"{doc.episode_id}｜{doc.title}｜發布 {doc.publish_date}｜"
            f"長度 {doc.duration_s}s｜{len(doc.segments)} 段"
        )

    def _expand_context(self, args: dict[str, Any]) -> tuple[str, list[RetrievedContext]]:
        doc = self._load(args["episode_id"])
        if not doc:
            return f"找不到 {args['episode_id']} 的逐字稿。", []
        w = float(args.get("window_s", 60))
        lo, hi = float(args["start_s"]) - w, float(args["end_s"]) + w
        segs = [s for s in doc.segments if s.end >= lo and s.start <= hi]
        text = "".join(s.text for s in segs)
        ctx = RetrievedContext(
            text=text,
            episode_id=doc.episode_id,
            ep_number=doc.ep_number,
            publish_date=doc.publish_date,
            start_s=segs[0].start if segs else lo,
            end_s=segs[-1].end if segs else hi,
            score=1.0,
            source="expand",
        )
        return text or "（該區間無逐字稿）", ([ctx] if segs else [])

    # --- helpers ---------------------------------------------------------------

    def _load(self, episode_id: str) -> TranscriptDoc | None:
        path = Path(self._s.transcripts_dir) / f"{episode_id}.json"
        return TranscriptDoc.load(path) if path.exists() else None


def _format_hits(hits: list[RetrievedContext]) -> str:
    if not hits:
        return "（沒有找到相關片段）"
    lines = []
    for h in hits:
        ts = f"{int((h.start_s or 0) // 60)}:{int((h.start_s or 0) % 60):02d}"
        lines.append(f"[{h.episode_id} @ {ts} | {h.publish_date}] {h.text}")
    return "\n\n".join(lines)
