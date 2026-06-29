"""Unit tests for corpus_range() — dependency-free (no network, no LLM, no mlx)."""

from __future__ import annotations

from types import SimpleNamespace

from app.ingestion.models import Segment, TranscriptDoc
from app.ingestion.status import corpus_range


def _write_doc(transcripts_dir, ep_number: int, publish_date: str) -> None:
    doc = TranscriptDoc(
        episode_id=f"EP{ep_number}",
        ep_number=ep_number,
        title=f"EP{ep_number} | 🌼",
        publish_date=publish_date,
        duration_s=1800,
        asr_provider="mlx",
        asr_model="whisper-large-v3",
        segments=[Segment(0.0, 1.0, "測試。")],
    )
    doc.save(transcripts_dir / f"{doc.episode_id}.json")


def test_corpus_range_summarizes_span(tmp_path):
    settings = SimpleNamespace(transcripts_dir=str(tmp_path))
    _write_doc(tmp_path, 100, "2026-01-05")
    _write_doc(tmp_path, 101, "2026-01-12")
    _write_doc(tmp_path, 102, "2026-01-19")

    r = corpus_range(settings)

    assert r.count == 3
    assert r.min_ep == 100
    assert r.max_ep == 102
    assert r.earliest_date == "2026-01-05"
    assert r.latest_date == "2026-01-19"
    assert r.human == "EP100–EP102 (3 episodes, 2026-01-05 … 2026-01-19)"


def test_corpus_range_empty(tmp_path):
    settings = SimpleNamespace(transcripts_dir=str(tmp_path))
    r = corpus_range(settings)
    assert r.count == 0
    assert r.min_ep is None
    assert r.max_ep is None
    assert r.earliest_date is None
    assert r.latest_date is None
    assert "0 episodes" in r.human


def test_corpus_range_single_episode(tmp_path):
    settings = SimpleNamespace(transcripts_dir=str(tmp_path))
    _write_doc(tmp_path, 671, "2026-06-17")
    r = corpus_range(settings)
    assert r.count == 1
    assert r.min_ep == r.max_ep == 671
    assert r.human == "EP671–EP671 (1 episode, 2026-06-17 … 2026-06-17)"
