"""BM25 keyword search over transcript chunks, jieba-tokenized for Chinese.

Loads all transcript JSONs, chunks them the same way the vector indexer does, and
builds an in-memory BM25 index. Fine for the test corpus (tens–hundreds of
episodes); swap for a server-side index if the catalog grows large.

Complements vector search in the agentic strategy: BM25 nails exact terms (ticker
symbols, names, numbers) that dense retrieval sometimes misses.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.chunker import chunk_segments
from app.ingestion.models import TranscriptDoc
from app.rag.base import RetrievedContext


def _tokenize(text: str) -> list[str]:
    import jieba  # lazy

    return [t for t in jieba.lcut(text) if t.strip()]


class KeywordIndex:
    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._bm25 = None
        self._meta: list[dict] = []  # parallel to corpus: payload per chunk

    def build(self) -> "KeywordIndex":
        from rank_bm25 import BM25Okapi  # lazy

        transcripts_dir = Path(self._s.transcripts_dir)
        corpus: list[list[str]] = []
        self._meta = []
        for path in sorted(transcripts_dir.glob("*.json")):
            doc = TranscriptDoc.load(path)
            for c in chunk_segments(
                doc.segments, self._s.chunk_target_chars, self._s.chunk_overlap_chars
            ):
                corpus.append(_tokenize(c.text))
                self._meta.append(
                    {
                        "text": c.text,
                        "episode_id": doc.episode_id,
                        "ep_number": doc.ep_number,
                        "publish_date": doc.publish_date,
                        "start_s": c.start_s,
                        "end_s": c.end_s,
                    }
                )
        self._bm25 = BM25Okapi(corpus) if corpus else None
        return self

    @property
    def size(self) -> int:
        return len(self._meta)

    def search(self, query: str, k: int = 8) -> list[RetrievedContext]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out: list[RetrievedContext] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            m = self._meta[i]
            out.append(
                RetrievedContext(
                    text=m["text"],
                    episode_id=m["episode_id"],
                    ep_number=m["ep_number"],
                    publish_date=m["publish_date"],
                    start_s=m["start_s"],
                    end_s=m["end_s"],
                    score=float(scores[i]),
                    source="keyword",
                )
            )
        return out
