"""Light cleanup of raw ASR segments before chunking/indexing.

Includes Simplified→Traditional conversion: Whisper tends to emit Simplified Chinese
even for this 繁體 (Taiwan) podcast, so we convert to Traditional (Taiwan standard)
for readability and exact keyword matching. Conversion is idempotent.
"""

from __future__ import annotations

import re

from app.ingestion.models import Segment

_WS = re.compile(r"[ \t]+")
_cc = None  # lazy OpenCC converter


def to_traditional(text: str) -> str:
    """Convert Simplified→Traditional (Taiwan). Returns input unchanged if OpenCC
    isn't installed, so this never hard-fails ingestion."""
    global _cc
    if _cc is None:
        try:
            from opencc import OpenCC  # lazy

            _cc = OpenCC("s2twp")  # Simplified → Traditional (Taiwan) w/ phrases
        except Exception:
            _cc = False
    if not _cc:
        return text
    try:
        return _cc.convert(text)
    except Exception:
        return text


def normalize_segments(segments: list[Segment], traditional: bool = True) -> list[Segment]:
    """Trim whitespace, drop empties, optionally convert to Traditional Chinese.
    Timestamps are preserved."""
    out: list[Segment] = []
    for s in segments:
        text = _WS.sub(" ", (s.text or "").strip())
        if not text:
            continue
        if traditional:
            text = to_traditional(text)
        out.append(Segment(start=s.start, end=s.end, text=text))
    return out
