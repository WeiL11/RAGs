"""Audio → timestamped transcript segments, with a pluggable engine.

Engines (set via ``asr_provider``):
  - "mlx"            : mlx-whisper, Apple-Silicon GPU. Free, local. Default.
  - "faster-whisper" : faster-whisper, CPU/CUDA. Free, local, portable.
  - "openai"         : OpenAI Whisper API. Paid (~$0.006/min), zero local setup.

Audio is downloaded to a temp file, transcribed, then **deleted** — we never keep
audio on disk. Engine libraries are imported lazily so the package imports without
them installed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx

from app.ingestion.models import Segment

# Whisper model size for local engines. "large-v3" = best quality (recommended for
# Mandarin); "large-v3-turbo" ~= large-v3 quality at much higher speed; "medium"/
# "small" are faster/lighter. Override via WHISPER_SIZE env, or point mlx at an exact
# HF repo with WHISPER_MLX_REPO (e.g. mlx-community/whisper-large-v3-turbo).
_LOCAL_WHISPER_SIZE = os.getenv("WHISPER_SIZE", "large-v3")
_MLX_REPO = os.getenv("WHISPER_MLX_REPO", f"mlx-community/whisper-{_LOCAL_WHISPER_SIZE}-mlx")


def download_audio(url: str, dest_dir: Path | None = None) -> Path:
    """Stream ``url`` to a temp .mp3 and return the path. Caller must delete it."""
    dest_dir = dest_dir or Path(tempfile.gettempdir())
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".mp3", dir=dest_dir)
    os.close(fd)
    path = Path(tmp)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for block in r.iter_bytes(chunk_size=1 << 16):
                f.write(block)
    return path


def transcribe_file(path: Path, provider: str, model: str, language: str = "zh") -> list[Segment]:
    if provider == "mlx":
        return _transcribe_mlx(path, language)
    if provider == "faster-whisper":
        return _transcribe_faster(path, language)
    if provider == "openai":
        return _transcribe_openai(path, model, language)
    raise ValueError(f"Unknown asr_provider: {provider!r}")


def transcribe_url(
    url: str, provider: str, model: str, language: str = "zh", keep_audio: bool = False
) -> list[Segment]:
    """Download → transcribe → delete audio. Returns timestamped segments."""
    path = download_audio(url)
    try:
        return transcribe_file(path, provider, model, language)
    finally:
        if not keep_audio:
            path.unlink(missing_ok=True)


# --- engines -------------------------------------------------------------------


def _transcribe_mlx(path: Path, language: str) -> list[Segment]:
    import mlx_whisper  # type: ignore

    result = mlx_whisper.transcribe(
        str(path), path_or_hf_repo=_MLX_REPO, language=language, word_timestamps=False
    )
    return [
        Segment(start=float(s["start"]), end=float(s["end"]), text=s["text"])
        for s in result.get("segments", [])
    ]


def _transcribe_faster(path: Path, language: str) -> list[Segment]:
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(_LOCAL_WHISPER_SIZE, device="auto", compute_type="auto")
    segments, _ = model.transcribe(str(path), language=language, vad_filter=True)
    return [Segment(start=float(s.start), end=float(s.end), text=s.text) for s in segments]


def _transcribe_openai(path: Path, model: str, language: str) -> list[Segment]:
    from openai import OpenAI  # type: ignore

    client = OpenAI()  # reads OPENAI_API_KEY
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    segs = getattr(resp, "segments", None) or []
    return [Segment(start=float(s.start), end=float(s.end), text=s.text) for s in segs]
