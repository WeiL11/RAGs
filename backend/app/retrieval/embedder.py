"""Pluggable text embedder. Same interface for ingestion and query time.

Providers (``embed_provider``):
  - "voyage" : voyage-3 (1024-d) — strong multilingual / Traditional Chinese. Default.
  - "openai" : text-embedding-3-large (3072-d).

``input_type`` distinguishes "document" (indexing) from "query" (search), which
Voyage uses to improve retrieval; OpenAI ignores it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings, get_settings


class Embedder(ABC):
    dim: int
    model: str

    @abstractmethod
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], input_type="query")[0]


class VoyageEmbedder(Embedder):
    # voyage-3 = 1024 dims; voyage-3-large = 1024 as well (configurable).
    _DIMS = {"voyage-3": 1024, "voyage-3-large": 1024, "voyage-3-lite": 512}

    def __init__(self, settings: Settings) -> None:
        import voyageai  # type: ignore

        self.model = settings.voyage_model
        self.dim = self._DIMS.get(self.model, 1024)
        self._client = voyageai.Client(api_key=settings.voyage_api_key or None)

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        # Voyage caps batch size; chunk to be safe.
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            r = self._client.embed(batch, model=self.model, input_type=input_type)
            out.extend(r.embeddings)
        return out


class OpenAIEmbedder(Embedder):
    _DIMS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # type: ignore

        self.model = settings.openai_embed_model
        self.dim = self._DIMS.get(self.model, 3072)
        self._client = OpenAI(api_key=settings.openai_api_key or None)

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            batch = texts[i : i + 256]
            r = self._client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in r.data])
        return out


class LocalEmbedder(Embedder):
    """Free, local embeddings via sentence-transformers (default BAAI/bge-m3, 1024-d).

    No API key, runs on the Mac (MPS/CPU). ``input_type`` is ignored. First use
    downloads the model (~2 GB)."""

    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self.model = settings.local_embed_model
        self._m = SentenceTransformer(self.model)
        get_dim = getattr(self._m, "get_embedding_dimension", None) or (
            self._m.get_sentence_embedding_dimension
        )
        self.dim = get_dim()

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        vecs = self._m.encode(
            texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embed_provider == "voyage":
        return VoyageEmbedder(settings)
    if settings.embed_provider == "openai":
        return OpenAIEmbedder(settings)
    if settings.embed_provider == "local":
        return LocalEmbedder(settings)
    raise ValueError(f"Unknown embed_provider: {settings.embed_provider!r}")
