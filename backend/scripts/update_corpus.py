"""Auto-update the transcript corpus and report its episode range — FREE for --status.

  python scripts/update_corpus.py --status     # just print the current range (no network)
  python scripts/update_corpus.py              # update with the newest episodes (default)
  python scripts/update_corpus.py --update     # same as default
  python scripts/update_corpus.py --limit 5    # only consider the 5 newest episodes

The update path reuses the idempotent ``ingest()`` — episodes already present are
re-used (no re-transcribe), only genuinely new ones are pulled from the RSS feed,
transcribed locally, and indexed. ``--status`` does no ingestion and needs no network.
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.ingestion.status import corpus_range


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Update the Gooaye RAG transcript corpus")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--status",
        action="store_true",
        help="print the current corpus range and exit (no ingestion, no network)",
    )
    mode.add_argument(
        "--update",
        action="store_true",
        help="pull the newest episodes and auto-update (default action)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="how many newest episodes to consider (default: settings.episode_window)",
    )
    args = p.parse_args(argv)

    settings = get_settings()

    if args.status:
        print(f"[corpus] {corpus_range(settings).human}")
        return 0

    # Default / --update: report, ingest the newest episodes, report again.
    print(f"[corpus] before: {corpus_range(settings).human}")
    limit = args.limit if args.limit is not None else settings.episode_window
    print(f"[update] running ingest(limit={limit}) — newest episodes only…")

    from app.ingestion.pipeline import ingest

    try:
        ingest(limit=limit)
    except KeyboardInterrupt:
        print("\n[abort] interrupted", file=sys.stderr)
        return 130

    print(f"[corpus] after:  {corpus_range(settings).human}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
