"""Turn a TranscriptDoc into chunks → embeddings → Qdrant points."""

from __future__ import annotations

import hashlib

from app.config import Settings
from app.ingestion.chunker import chunk_segments
from app.ingestion.models import TranscriptDoc
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore


def _point_id(episode_id: str, chunk_index: int) -> int:
    # Stable 63-bit id from episode + chunk so re-indexing upserts in place.
    h = hashlib.sha1(f"{episode_id}:{chunk_index}".encode()).hexdigest()
    return int(h[:15], 16)


def index_document(
    doc: TranscriptDoc, embedder: Embedder, store: VectorStore, settings: Settings
) -> int:
    """Chunk, embed, and upsert one episode. Returns the number of chunks indexed."""
    chunks = chunk_segments(
        doc.segments,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    if not chunks:
        return 0

    vectors = embedder.embed([c.text for c in chunks], input_type="document")
    points = [
        {
            "id": _point_id(doc.episode_id, c.chunk_index),
            "text": c.text,
            "episode_id": doc.episode_id,
            "ep_number": doc.ep_number,
            "publish_date": doc.publish_date,
            "start_s": c.start_s,
            "end_s": c.end_s,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    store.upsert(points, vectors)
    return len(chunks)
