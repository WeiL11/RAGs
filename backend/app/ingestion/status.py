"""Corpus status — dependency-free reporting over the on-disk transcript set.

The canonical corpus is one ``TranscriptDoc`` JSON per episode under
``settings.transcripts_dir``. ``corpus_range()`` scans those files and summarizes
the episode-number and publish-date span. This is intentionally free and offline:
no network, no LLM, no ASR/vector deps — so it can answer "what do we have?" cheaply
(e.g. before/after an auto-update). Heavy imports are done lazily inside the function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorpusRange:
    """A small summary of the transcript corpus on disk."""

    count: int
    min_ep: int | None
    max_ep: int | None
    earliest_date: str | None
    latest_date: str | None
    human: str


def corpus_range(settings=None) -> CorpusRange:  # noqa: ANN001
    """Scan ``settings.transcripts_dir`` for transcript JSONs and summarize the span.

    Returns a :class:`CorpusRange`. Handles an empty/missing corpus gracefully
    (count 0, ``None`` fields, and an explanatory ``human`` string). Does NOT need
    the network or an LLM.
    """
    # Lazy imports keep this module import-safe and free of heavy/IO deps at top level.
    from pathlib import Path

    from app.config import get_settings
    from app.ingestion.models import TranscriptDoc

    if settings is None:
        settings = get_settings()

    transcripts_dir = Path(settings.transcripts_dir)
    paths = sorted(transcripts_dir.glob("*.json")) if transcripts_dir.exists() else []

    docs: list[TranscriptDoc] = []
    for p in paths:
        try:
            docs.append(TranscriptDoc.load(p))
        except (ValueError, KeyError, TypeError):
            # Skip anything that isn't a well-formed transcript JSON.
            continue

    if not docs:
        return CorpusRange(
            count=0,
            min_ep=None,
            max_ep=None,
            earliest_date=None,
            latest_date=None,
            human="empty corpus (0 episodes)",
        )

    eps = [d.ep_number for d in docs]
    min_ep, max_ep = min(eps), max(eps)
    dates = sorted(d.publish_date for d in docs if d.publish_date)
    earliest_date = dates[0] if dates else None
    latest_date = dates[-1] if dates else None

    date_part = (
        f", {earliest_date} … {latest_date}" if earliest_date and latest_date else ""
    )
    human = (
        f"EP{min_ep}–EP{max_ep} ({len(docs)} episode"
        f"{'s' if len(docs) != 1 else ''}{date_part})"
    )

    return CorpusRange(
        count=len(docs),
        min_ep=min_ep,
        max_ep=max_ep,
        earliest_date=earliest_date,
        latest_date=latest_date,
        human=human,
    )
