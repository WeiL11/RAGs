"""Application settings.

All knobs live here so iterating on the RAG layer is config, not code surgery.
Values are read from environment variables / a local ``.env`` file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# backend/  (this file is backend/app/config.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent
# repo root (gooaye-rag/)
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App shell ---
    app_name: str = "Gooaye RAG"
    # API security. If api_auth_token is set, /chat and /strategies require it via
    # `Authorization: Bearer <token>` or `X-API-Key`. Empty = open (dev only — the
    # app warns at startup). rate_limit_per_min = per-client-IP cap (0 disables).
    api_auth_token: str = ""
    rate_limit_per_min: int = 120
    # NoDecode: take the raw env string (CSV) and let _split_csv parse it, instead of
    # pydantic-settings trying to JSON-decode these list fields.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    # Which strategies to expose; first is the UI default. Add "agentic"/"graph"
    # once their milestones land.
    enabled_strategies: Annotated[list[str], NoDecode] = [
        "echo",
        "agentic",
        "graph",
        "corrective",
    ]

    # --- Generation provider ---
    # "anthropic" (Claude, paid) or "gemini" (free tier). The free HF Spaces deploy
    # uses gemini; local dev defaults to anthropic.
    llm_provider: str = "anthropic"

    # Claude
    anthropic_api_key: str = ""
    answer_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"  # RAGAS judge (M4)

    # Gemini (free tier — Google AI Studio key). 2.0-flash has a much higher free
    # daily quota than 2.5-flash (~20/day). Override with GEMINI_MODEL.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    max_tokens: int = 4096
    agent_max_iterations: int = 6  # cap tool-use rounds per query

    # Corrective RAG (M-extra): cheap model grades retrieved passages; on weak
    # results, rewrite the query and re-retrieve up to crag_max_rounds times.
    grader_model: str = "claude-haiku-4-5"
    crag_k: int = 8
    crag_max_rounds: int = 3

    # --- Embeddings ---
    embed_provider: str = "voyage"  # "voyage" | "openai" | "local"
    voyage_api_key: str = ""
    voyage_model: str = "voyage-3"
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-large"
    # Local embeddings (free, no key) — BGE-M3 is strong on Traditional Chinese.
    local_embed_model: str = "BAAI/bge-m3"

    # --- ASR (M1) ---
    # Default to local mlx-whisper (free, Apple Silicon). "openai" = paid Whisper API,
    # "faster-whisper" = local CPU/CUDA. asr_model only applies to the openai engine.
    asr_provider: str = "mlx"
    asr_model: str = "whisper-1"

    # --- Vector store ---
    # "local"  = embedded Qdrant, persisted under qdrant_path (no Docker; great for
    #            the test corpus). "server" = connect to qdrant_url (for deployment).
    qdrant_mode: str = "local"
    qdrant_path: str = str(DATA_DIR / "qdrant_local")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "gooaye_chunks"

    # --- Graph store (M3) ---
    graph_path: str = str(DATA_DIR / "graph.json")
    # Entity/relation extraction model (cheap + structured → Haiku by default).
    extract_model: str = "claude-haiku-4-5"

    # --- Ingestion (M1) ---
    rss_url: str = (
        "https://feeds.soundon.fm/podcasts/"
        "954689a5-3096-43a4-a80b-7810b219cef3.xml"
    )
    # Cap how many (most recent) episodes to ingest — keeps device footprint small.
    # Set to 0 to ingest the entire back-catalog.
    episode_window: int = 30
    transcripts_dir: str = str(DATA_DIR / "transcripts")

    # Chunking
    chunk_target_chars: int = 700  # CJK-aware target chunk size
    chunk_overlap_chars: int = 120


    # Accept comma-separated env values for list fields (friendlier than JSON arrays),
    # e.g. ENABLED_STRATEGIES=echo,agentic,graph
    @field_validator("enabled_strategies", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


def _resolve_secret(value: str, file_env: str) -> str:
    """Secrets-manager-friendly: if a key is empty but <NAME>_FILE points to a file
    (Docker/K8s secrets convention), read the secret from that file."""
    if value:
        return value
    path = os.getenv(file_env)
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8").strip()
    return value


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # Support *_FILE secret sources without hardcoding keys.
    s.anthropic_api_key = _resolve_secret(s.anthropic_api_key, "ANTHROPIC_API_KEY_FILE")
    s.gemini_api_key = _resolve_secret(s.gemini_api_key, "GEMINI_API_KEY_FILE")
    s.voyage_api_key = _resolve_secret(s.voyage_api_key, "VOYAGE_API_KEY_FILE")
    s.openai_api_key = _resolve_secret(s.openai_api_key, "OPENAI_API_KEY_FILE")
    s.api_auth_token = _resolve_secret(s.api_auth_token, "API_AUTH_TOKEN_FILE")
    return s
