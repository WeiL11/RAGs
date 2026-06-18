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
        from app.rag.graph.extract import ClaudeExtractor

        extractor = ClaudeExtractor(settings)
    store = store or GraphStore(settings.graph_path)

    paths = sorted(
        Path(settings.transcripts_dir).glob("*.json"),
        key=lambda p: TranscriptDoc.load(p).ep_number,
        reverse=True,
    )
    if limit:
        paths = paths[:limit]

    for path in paths:
        doc = TranscriptDoc.load(path)
        chunks = chunk_segments(
            doc.segments, settings.chunk_target_chars, settings.chunk_overlap_chars
        )
        print(f"[graph] {doc.episode_id}: {len(chunks)} chunks", flush=True)
        for c in chunks:
            result = extractor.extract(c.text)
            mention = {
                "text": c.text,
                "episode_id": doc.episode_id,
                "ep_number": doc.ep_number,
                "publish_date": doc.publish_date,
                "start_s": c.start_s,
                "end_s": c.end_s,
            }
            for ent in result.get("entities", []):
                store.add_entity(ent["name"], ent.get("type", "other"), mention)
            for rel in result.get("relations", []):
                # ensure endpoints exist even if not listed as entities
                store.add_entity(rel["subject"], "other")
                store.add_entity(rel["object"], "other")
                store.add_relation(
                    rel["subject"], rel["relation"], rel["object"], doc.episode_id, c.text
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
