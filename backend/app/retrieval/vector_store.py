"""Qdrant wrapper. Shared by the ingestion indexer and the RAG strategies.

Payload schema per point (everything needed to build a ``RetrievedContext``):
    text, episode_id, ep_number, publish_date, start_s, end_s, chunk_index
"""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.rag.base import RetrievedContext

# Fields that must match between the index that was BUILT and the config QUERYING it.
# Critically includes embed_model, not just dim: voyage-3 and BGE-M3 are both 1024-d,
# so a dim-only check would let you silently search one vector space with another.
_MANIFEST_FIELDS = (
    "embed_provider",
    "embed_model",
    "dim",
    "chunk_target_chars",
    "chunk_overlap_chars",
)


def build_manifest(settings: Settings, embedder) -> dict[str, Any]:  # noqa: ANN001
    """Identity of the index: which model/params produced these vectors."""
    return {
        "embed_provider": settings.embed_provider,
        "embed_model": embedder.model,
        "dim": embedder.dim,
        "chunk_target_chars": settings.chunk_target_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
    }


class IndexMismatchError(RuntimeError):
    """Raised when the vector index was built with different embedding/chunk config."""


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        from qdrant_client import QdrantClient  # type: ignore

        self._s = settings or get_settings()
        if self._s.qdrant_mode == "local":
            # Embedded, on-disk Qdrant — no Docker. Use ":memory:" path for tests.
            loc = self._s.qdrant_path
            self._client = (
                QdrantClient(location=":memory:")
                if loc == ":memory:"
                else QdrantClient(path=loc)
            )
        else:
            self._client = QdrantClient(url=self._s.qdrant_url)
        self.collection = self._s.qdrant_collection

    def ensure_collection(self, dim: int, manifest: dict[str, Any] | None = None) -> None:
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        if manifest is not None:
            stored = self._read_manifest()
            if stored is None:
                self._write_manifest(manifest)
            else:
                self._assert_match(stored, manifest)

    # --- index-version guard ---------------------------------------------------

    def _manifest_collection(self) -> str:
        return f"{self.collection}_manifest"

    def _read_manifest(self) -> dict[str, Any] | None:
        try:
            pts = self._client.retrieve(
                collection_name=self._manifest_collection(), ids=[1], with_payload=True
            )
        except Exception:
            return None
        return (pts[0].payload or {}) if pts else None

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams  # type: ignore

        mc = self._manifest_collection()
        existing = {c.name for c in self._client.get_collections().collections}
        if mc not in existing:
            self._client.create_collection(
                collection_name=mc, vectors_config=VectorParams(size=1, distance=Distance.DOT)
            )
        self._client.upsert(
            collection_name=mc, points=[PointStruct(id=1, vector=[1.0], payload=manifest)]
        )

    def assert_manifest(self, manifest: dict[str, Any]) -> None:
        """Verify the live index was built with the given config; raise if not."""
        stored = self._read_manifest()
        if stored is None:
            raise IndexMismatchError(
                "Vector index has no manifest (built before the version guard, or empty). "
                "Rebuild it: `python -m app.ingestion.pipeline --reindex`."
            )
        self._assert_match(stored, manifest)

    @staticmethod
    def _assert_match(stored: dict[str, Any], current: dict[str, Any]) -> None:
        diff = [k for k in _MANIFEST_FIELDS if stored.get(k) != current.get(k)]
        if diff:
            raise IndexMismatchError(
                "Embedding/index mismatch — the vector index was built with "
                f"{ {k: stored.get(k) for k in _MANIFEST_FIELDS} }, but current config is "
                f"{ {k: current.get(k) for k in _MANIFEST_FIELDS} }. Differs on: {diff}. "
                "Rebuild with `--reindex` or restore the matching settings."
            )

    def upsert(self, points: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        """``points`` are payload dicts (must include a stable int/str ``id``)."""
        from qdrant_client.models import PointStruct  # type: ignore

        structs = [
            PointStruct(id=p["id"], vector=v, payload={k: val for k, val in p.items() if k != "id"})
            for p, v in zip(points, vectors)
        ]
        self._client.upsert(collection_name=self.collection, points=structs)

    def has_episode(self, episode_id: str) -> bool:
        from qdrant_client.models import (  # type: ignore
            FieldCondition,
            Filter,
            MatchValue,
        )

        res = self._client.count(
            collection_name=self.collection,
            count_filter=Filter(
                must=[FieldCondition(key="episode_id", match=MatchValue(value=episode_id))]
            ),
            exact=True,
        )
        return res.count > 0

    def search(
        self, vector: list[float], k: int = 8, filters: dict[str, Any] | None = None
    ) -> list[RetrievedContext]:
        qfilter = self._build_filter(filters)
        response = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            query_filter=qfilter,
            with_payload=True,
        )
        out: list[RetrievedContext] = []
        for h in response.points:
            p = h.payload or {}
            out.append(
                RetrievedContext(
                    text=p.get("text", ""),
                    episode_id=p.get("episode_id", ""),
                    ep_number=int(p.get("ep_number", 0)),
                    publish_date=p.get("publish_date"),
                    start_s=p.get("start_s"),
                    end_s=p.get("end_s"),
                    score=float(h.score),
                    source="vector",
                )
            )
        return out

    def _build_filter(self, filters: dict[str, Any] | None):
        """Supports {"date_from": "YYYY-MM-DD", "date_to": ..., "ep_min": int, ...}."""
        if not filters:
            return None
        from qdrant_client.models import (  # type: ignore
            DatetimeRange,
            FieldCondition,
            Filter,
            Range,
        )

        must = []
        if filters.get("date_from") or filters.get("date_to"):
            must.append(
                FieldCondition(
                    key="publish_date",
                    range=DatetimeRange(gte=filters.get("date_from"), lte=filters.get("date_to")),
                )
            )
        if filters.get("ep_min") is not None or filters.get("ep_max") is not None:
            must.append(
                FieldCondition(
                    key="ep_number",
                    range=Range(gte=filters.get("ep_min"), lte=filters.get("ep_max")),
                )
            )
        return Filter(must=must) if must else None
