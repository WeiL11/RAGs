"""Canonical data types shared across the ingestion pipeline.

The on-disk source of truth is one ``TranscriptDoc`` JSON file per episode under
``data/transcripts/``. Chunks are derived from it at index time and stored in the
vector DB; we keep transcripts (not audio) so re-chunking/re-embedding is cheap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Episode:
    """Metadata for one podcast episode, parsed from the RSS feed."""

    episode_id: str  # e.g. "EP512" (falls back to a slug if no number found)
    ep_number: int  # 0 if not parseable
    title: str
    audio_url: str
    publish_date: str | None  # ISO-8601 date
    duration_s: int | None


@dataclass
class Segment:
    """A timestamped span of transcript, as emitted by the ASR engine."""

    start: float
    end: float
    text: str


@dataclass
class Chunk:
    """A retrieval unit: contiguous segments grouped to a target size."""

    chunk_index: int
    text: str
    start_s: float
    end_s: float


@dataclass
class TranscriptDoc:
    """Full canonical transcript for one episode (the source of truth)."""

    episode_id: str
    ep_number: int
    title: str
    publish_date: str | None
    duration_s: int | None
    asr_provider: str
    asr_model: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "".join(s.text for s in self.segments)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscriptDoc":
        segs = [Segment(**s) for s in d.get("segments", [])]
        return cls(**{**d, "segments": segs})

    @classmethod
    def load(cls, path: Path) -> "TranscriptDoc":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
