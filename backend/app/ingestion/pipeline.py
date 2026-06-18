"""End-to-end ingestion orchestrator.

  RSS → (per episode) download audio → ASR → normalize → save transcript JSON
       → chunk → embed → upsert to Qdrant → delete audio

Idempotent: an episode whose transcript JSON already exists is re-used (ASR is
skipped); pass ``--reindex`` to re-chunk/re-embed without re-transcribing, or
``--force`` to redo ASR too. Honors ``settings.episode_window`` to cap how many
(newest) episodes are processed — keeping the device footprint small.

Run:
    python -m app.ingestion.pipeline                 # use config window
    python -m app.ingestion.pipeline --limit 5       # just the 5 newest
    python -m app.ingestion.pipeline --transcribe-only   # skip vector indexing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.ingestion.asr import transcribe_url
from app.ingestion.models import Episode, TranscriptDoc
from app.ingestion.normalize import normalize_segments
from app.ingestion.rss import fetch_episodes


def _transcript_path(transcripts_dir: Path, ep: Episode) -> Path:
    return transcripts_dir / f"{ep.episode_id}.json"


def ingest(
    limit: int | None = None,
    *,
    transcribe_only: bool = False,
    reindex: bool = False,
    force: bool = False,
) -> None:
    settings = get_settings()
    transcripts_dir = Path(settings.transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    window = limit if limit is not None else settings.episode_window
    episodes = fetch_episodes(settings.rss_url, window=window)
    print(f"[rss] {len(episodes)} episode(s) to consider (window={window or 'all'})")

    # Lazy: only construct embedder/store if we actually index.
    embedder = store = None
    if not transcribe_only:
        from app.ingestion.index_vectors import index_document
        from app.retrieval.embedder import get_embedder
        from app.retrieval.vector_store import VectorStore, build_manifest

        embedder = get_embedder(settings)
        store = VectorStore(settings)
        store.ensure_collection(embedder.dim, build_manifest(settings, embedder))
        print(f"[index] embedder={embedder.model} dim={embedder.dim} -> {store.collection}")

    for ep in episodes:
        tpath = _transcript_path(transcripts_dir, ep)
        t0 = time.perf_counter()

        # 1) transcript (re-use if present unless forced)
        if tpath.exists() and not force:
            doc = TranscriptDoc.load(tpath)
            stage = "cached"
        else:
            print(f"[asr ] {ep.episode_id} transcribing ({settings.asr_provider})…", flush=True)
            segments = normalize_segments(
                transcribe_url(ep.audio_url, settings.asr_provider, settings.asr_model)
            )
            doc = TranscriptDoc(
                episode_id=ep.episode_id,
                ep_number=ep.ep_number,
                title=ep.title,
                publish_date=ep.publish_date,
                duration_s=ep.duration_s,
                asr_provider=settings.asr_provider,
                asr_model=settings.asr_model,
                segments=segments,
            )
            doc.save(tpath)
            stage = "transcribed"

        # 2) index (unless transcribe-only or already indexed)
        n_chunks = 0
        if not transcribe_only:
            assert embedder and store
            already = store.has_episode(ep.episode_id)
            if already and not (reindex or force):
                stage += ", indexed(skip)"
            else:
                n_chunks = index_document(doc, embedder, store, settings)
                stage += f", indexed {n_chunks} chunks"

        dt = time.perf_counter() - t0
        print(
            f"[done] {ep.episode_id} {ep.publish_date}  "
            f"{len(doc.segments)} segs  {stage}  ({dt:.1f}s)"
        )

    print("[ok] ingestion complete")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gooaye RAG ingestion pipeline")
    p.add_argument("--limit", type=int, default=None, help="override episode window")
    p.add_argument("--transcribe-only", action="store_true", help="skip vector indexing")
    p.add_argument("--reindex", action="store_true", help="re-embed existing transcripts")
    p.add_argument("--force", action="store_true", help="re-transcribe even if cached")
    args = p.parse_args(argv)
    try:
        ingest(
            limit=args.limit,
            transcribe_only=args.transcribe_only,
            reindex=args.reindex,
            force=args.force,
        )
    except KeyboardInterrupt:
        print("\n[abort] interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
