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

    # Extract ONCE per episode (free-tier friendly: ~1 request/episode instead of
    # hundreds). Mention/relation provenance is recovered by matching entity names to
    # the chunk they appear in, so we keep chunk-level timestamps.
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

        result = extractor.extract(doc.full_text[:20000])
        ents = result.get("entities", [])
        rels = result.get("relations", [])
        print(f"[graph] {doc.episode_id}: {len(ents)} entities, {len(rels)} relations", flush=True)

        for ent in ents:
            store.add_entity(ent["name"], ent.get("type", "other"), _mention(_chunk_for(ent["name"])))
        for rel in rels:
            store.add_entity(rel["subject"], "other")
            store.add_entity(rel["object"], "other")
            c = _chunk_for(rel["subject"]) or _chunk_for(rel["object"])
            store.add_relation(
                rel["subject"], rel["relation"], rel["object"], doc.episode_id, c.text if c else ""
            )

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
