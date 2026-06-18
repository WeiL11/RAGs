"""CJK-aware chunking.

Chinese has no word spaces, so we chunk by *character count* on sentence
boundaries (。！？…；newlines) rather than by whitespace tokens. Each chunk carries
the timestamp range of the segments it spans, so retrieval can deep-link to audio.
Adjacent chunks overlap by ``overlap_chars`` to avoid splitting context.
"""

from __future__ import annotations

import re

from app.ingestion.models import Chunk, Segment

# Sentence-ending punctuation (full-width CJK + ASCII fallback).
_SENT_END = re.compile(r"(?<=[。！？…；!?;\n])")


def _split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENT_END.split(text) if p.strip()]
    return parts or ([text] if text.strip() else [])


def chunk_segments(
    segments: list[Segment], target_chars: int = 700, overlap_chars: int = 120
) -> list[Chunk]:
    """Group consecutive segments into ~``target_chars`` chunks on sentence ends.

    We attribute each sentence to the timestamp of the segment it came from, so a
    chunk's ``start_s``/``end_s`` bracket its real audio span. Overlap is applied by
    carrying the tail of the previous chunk's text into the next.
    """
    # Flatten into (sentence, start, end) keeping timing from the source segment.
    sentences: list[tuple[str, float, float]] = []
    for seg in segments:
        for sent in _split_sentences(seg.text):
            sentences.append((sent, seg.start, seg.end))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    start_s: float | None = None
    end_s = 0.0
    idx = 0

    def flush() -> None:
        nonlocal buf, buf_len, start_s, idx
        if not buf:
            return
        text = "".join(buf).strip()
        if text:
            chunks.append(
                Chunk(chunk_index=idx, text=text, start_s=start_s or 0.0, end_s=end_s)
            )
            idx += 1
        # seed next buffer with overlap tail
        tail = text[-overlap_chars:] if overlap_chars else ""
        buf = [tail] if tail else []
        buf_len = len(tail)
        start_s = None

    for sent, s_start, s_end in sentences:
        if start_s is None:
            start_s = s_start
        buf.append(sent)
        buf_len += len(sent)
        end_s = s_end
        if buf_len >= target_chars:
            flush()

    flush()
    # Re-index in case overlap-seeded empties shifted things.
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks
