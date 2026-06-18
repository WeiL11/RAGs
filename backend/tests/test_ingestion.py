"""Unit tests for the dependency-free ingestion logic (chunker, normalize, rss parse)."""

from __future__ import annotations

from app.ingestion.chunker import chunk_segments
from app.ingestion.models import Segment, TranscriptDoc
from app.ingestion.normalize import normalize_segments
from app.ingestion.rss import _parse_ep_number


def test_parse_ep_number():
    assert _parse_ep_number("EP512 | 🌼") == 512
    assert _parse_ep_number("第 67 集 聊聊") == 67
    assert _parse_ep_number("無編號特別篇") == 0


def test_normalize_drops_empty_and_trims():
    segs = [Segment(0, 1, "  你好   世界 "), Segment(1, 2, "   "), Segment(2, 3, "再見")]
    out = normalize_segments(segs)
    assert [s.text for s in out] == ["你好 世界", "再見"]


def test_chunker_respects_target_and_carries_timestamps():
    # Build long CJK text across several timestamped segments.
    segs = [
        Segment(start=i * 10.0, end=i * 10.0 + 10.0, text="這是一個測試句子。" * 5)
        for i in range(6)
    ]
    chunks = chunk_segments(segs, target_chars=200, overlap_chars=20)
    assert len(chunks) >= 2
    # chunks are contiguous & timestamped within the source range
    assert chunks[0].start_s == 0.0
    assert chunks[-1].end_s == 60.0
    assert all(c.end_s >= c.start_s for c in chunks)
    # indices are sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_handles_single_short_segment():
    chunks = chunk_segments([Segment(0, 5, "短句。")], target_chars=700, overlap_chars=50)
    assert len(chunks) == 1
    assert chunks[0].text == "短句。"


def test_transcript_roundtrip(tmp_path):
    doc = TranscriptDoc(
        episode_id="EP1",
        ep_number=1,
        title="EP1 | 測試",
        publish_date="2024-01-01",
        duration_s=3000,
        asr_provider="mlx",
        asr_model="large-v3",
        segments=[Segment(0, 2, "你好"), Segment(2, 4, "世界")],
    )
    path = tmp_path / "EP1.json"
    doc.save(path)
    loaded = TranscriptDoc.load(path)
    assert loaded.full_text == "你好世界"
    assert loaded.segments[1].end == 4
