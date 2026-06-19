"""Build the knowledge graph from saved transcripts (M3, ingest-time).

  transcripts/*.json → chunk → LLM extract entities+relations → GraphStore → graph.json

Run:
    python -m app.ingestion.build_graph                # all transcripts
    python -m app.ingestion.build_graph --limit 5      # newest 5 (by ep number)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.chunker import chunk_segments
from app.ingestion.models import TranscriptDoc
from app.rag.graph.extract import GraphExtractor
from app.retrieval.graph_store import GraphStore


def build_graph(
    settings: Settings | None = None,
    *,
    extractor: GraphExtractor | None = None,
    store: GraphStore | None = None,
    limit: int | None = None,
) -> GraphStore:
    settings = settings or get_settings()
    if extractor is None:
        from app.rag.graph.extract import get_extractor

        extractor = get_extractor(settings)
    store = store or GraphStore(settings.graph_path)

    paths = sorted(
        Path(settings.transcripts_dir).glob("*.json"),
        key=lambda p: TranscriptDoc.load(p).ep_number,
        reverse=True,
    )
    if limit:
        paths = paths[:limit]

    # Window each episode to fit the LLM's tokens-per-minute limit (Groq free tier ≈ 12k
    # TPM → a full ~20k-char episode is too big for one request). Extract per window,
    # throttle GLOBALLY between calls, merge + de-dupe. Mention provenance is recovered by
    # matching entity names to the chunk they appear in.
    import os
    import time

    win = int(os.getenv("GRAPH_WINDOW_CHARS", "6000"))
    max_win = int(os.getenv("GRAPH_MAX_WINDOWS", "2"))
    throttle = float(os.getenv("GRAPH_THROTTLE_S", "33"))
    calls = 0

    for path in paths:
        doc = TranscriptDoc.load(path)
        chunks = chunk_segments(
            doc.segments, settings.chunk_target_chars, settings.chunk_overlap_chars
        )

        def _chunk_for(name: str):
            for c in chunks:
                if name and name in c.text:
                    return c
            return None

        def _mention(c) -> dict | None:
            if c is None:
                return None
            return {
                "text": c.text,
                "episode_id": doc.episode_id,
                "ep_number": doc.ep_number,
                "publish_date": doc.publish_date,
                "start_s": c.start_s,
                "end_s": c.end_s,
            }

        text = doc.full_text
        windows = [text[i : i + win] for i in range(0, len(text), win)][:max_win] or [text]

        seen_e: set[str] = set()
        seen_r: set[tuple] = set()
        ents: list[dict] = []
        rels: list[dict] = []
        for w in windows:
            if not w.strip():
                continue
            if calls:
                time.sleep(throttle)
            calls += 1
            try:
                r = extractor.extract(w)
            except Exception as exc:  # noqa: BLE001 — quota/rate errors: skip, keep the rest
                print(f"[skip] {doc.episode_id}: {str(exc)[:120]}", flush=True)
                break
            for e in r.get("entities", []):
                k = (e.get("name") or "").strip().lower()
                if k and k not in seen_e:
                    seen_e.add(k)
                    ents.append(e)
            for rel in r.get("relations", []):
                k = (rel.get("subject", ""), rel.get("relation", ""), rel.get("object", ""))
                if all(k) and k not in seen_r:
                    seen_r.add(k)
                    rels.append(rel)
        print(f"[graph] {doc.episode_id}: {len(ents)} entities, {len(rels)} relations "
              f"({len(windows)} window(s))", flush=True)

        for ent in ents:
            store.add_entity(ent["name"], ent.get("type", "other"), _mention(_chunk_for(ent["name"])))
        for rel in rels:
            store.add_entity(rel["subject"], "other")
            store.add_entity(rel["object"], "other")
            c = _chunk_for(rel["subject"]) or _chunk_for(rel["object"])
            store.add_relation(
                rel["subject"], rel["relation"], rel["object"], doc.episode_id, c.text if c else ""
            )
        store.save()  # incremental: persist after every episode so a later quota error can't lose progress

    store.save()
    print(f"[ok] graph: {store.num_nodes} nodes, {store.num_edges} edges → {store.path}")
    return store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Graph RAG knowledge graph")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    build_graph(limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
