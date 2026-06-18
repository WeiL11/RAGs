"""Parse the Gooaye podcast RSS feed into ``Episode`` records.

The feed (SoundOn) carries CDATA-wrapped titles like ``EP512 | ...``; ``feedparser``
unwraps these for us. Episode numbers are extracted from the title.
"""

from __future__ import annotations

import re
from datetime import datetime

import feedparser

from app.ingestion.models import Episode

# Matches "EP512", "ep 512", "第512集", etc. at/near the start of a title.
_EP_RE = re.compile(r"(?:EP|ep|第)\s*0*(\d{1,4})")


def _parse_ep_number(title: str) -> int:
    m = _EP_RE.search(title)
    return int(m.group(1)) if m else 0


def _parse_date(entry: object) -> str | None:
    t = getattr(entry, "published_parsed", None)
    if not t:
        return None
    return datetime(*t[:6]).date().isoformat()


def _audio_url(entry: object) -> str | None:
    for enc in getattr(entry, "enclosures", []) or []:
        href = enc.get("href") or enc.get("url")
        if href:
            return href
    # some feeds use links rel="enclosure"
    for link in getattr(entry, "links", []) or []:
        if link.get("rel") == "enclosure" and link.get("href"):
            return link["href"]
    return None


def _duration_s(entry: object) -> int | None:
    raw = entry.get("itunes_duration") if hasattr(entry, "get") else None
    if not raw:
        return None
    raw = str(raw).strip()
    if raw.isdigit():
        return int(raw)
    # HH:MM:SS or MM:SS
    parts = raw.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    sec = 0
    for n in nums:
        sec = sec * 60 + n
    return sec


def fetch_episodes(rss_url: str, window: int = 0) -> list[Episode]:
    """Return episodes newest-first. ``window`` caps the count (0 = all).

    Episodes without a downloadable audio URL are skipped.
    """
    feed = feedparser.parse(rss_url)
    episodes: list[Episode] = []
    for entry in feed.entries:
        title = (getattr(entry, "title", "") or "").strip()
        audio = _audio_url(entry)
        if not audio:
            continue
        ep_num = _parse_ep_number(title)
        episodes.append(
            Episode(
                episode_id=f"EP{ep_num}" if ep_num else _slug(title),
                ep_number=ep_num,
                title=title,
                audio_url=audio,
                publish_date=_parse_date(entry),
                duration_s=_duration_s(entry),
            )
        )
    # Feed is already newest-first, but sort defensively by ep number desc.
    episodes.sort(key=lambda e: e.ep_number, reverse=True)
    return episodes[:window] if window else episodes


def _slug(title: str) -> str:
    s = re.sub(r"\s+", "-", title.strip())[:40]
    return s or "EP-unknown"
