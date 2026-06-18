"""Convert existing transcripts Simplified→Traditional (Taiwan) in place.

One-off maintenance: applies OpenCC s2twp to every saved transcript's segment text,
so transcripts produced before the normalizer included conversion become Traditional.
Idempotent — safe to run repeatedly. No re-transcription, free, local.

    python scripts/to_traditional.py
"""

from __future__ import annotations

import glob
from pathlib import Path

from app.config import get_settings
from app.ingestion.models import TranscriptDoc
from app.ingestion.normalize import to_traditional


def main() -> None:
    s = get_settings()
    files = sorted(glob.glob(f"{s.transcripts_dir}/*.json"))
    if not files:
        print(f"no transcripts in {s.transcripts_dir}")
        return
    for f in files:
        doc = TranscriptDoc.load(Path(f))
        changed = 0
        for seg in doc.segments:
            t = to_traditional(seg.text)
            if t != seg.text:
                seg.text = t
                changed += 1
        doc.save(Path(f))
        print(f"  {doc.episode_id}: converted {changed}/{len(doc.segments)} segments")
    print(f"[ok] {len(files)} transcripts now Traditional. Re-run demo_search to see it.")
    print("     (To refresh the Qdrant vector index too: "
          "python -m app.ingestion.pipeline --limit 14 --reindex)")


if __name__ == "__main__":
    main()
