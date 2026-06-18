"""Demo: how retrieval works — semantic search + automatic similar-word discovery
+ keyword (BM25) search. Runs fully local (BGE-M3 + cosine), no API key.

Usage:
    python scripts/demo_search.py "諾亞"
    python scripts/demo_search.py "諾亞" --dir /tmp/gooaye_smoke

This is a teaching/inspection tool, not part of the app. It loads transcripts,
embeds their chunks in memory, and shows what each search method returns.
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.ingestion.chunker import chunk_segments
from app.ingestion.models import TranscriptDoc


def _fmt_ts(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--dir", default=None, help="transcripts dir (default: data, fallback smoke)")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    s = get_settings()
    s.embed_provider = "local"
    tdir = args.dir or s.transcripts_dir
    if not glob.glob(f"{tdir}/*.json"):
        tdir = "/tmp/gooaye_smoke"
    files = sorted(glob.glob(f"{tdir}/*.json"))
    print(f"corpus: {len(files)} episode(s) from {tdir}\n")

    # 1) chunk all transcripts
    chunks: list[tuple[TranscriptDoc, object]] = []
    for f in files:
        doc = TranscriptDoc.load(Path(f))
        for c in chunk_segments(doc.segments, s.chunk_target_chars, s.chunk_overlap_chars):
            chunks.append((doc, c))
    texts = [c.text for _, c in chunks]
    print(f"chunks: {len(texts)}")

    # 2) embed chunks + query with the SAME local model the app uses
    from app.retrieval.embedder import get_embedder

    emb = get_embedder(s)
    print(f"embedder: {emb.model} ({emb.dim}-d)\n")
    M = np.array(emb.embed(texts), dtype="float32")  # normalized
    q = np.array(emb.embed_query(args.query), dtype="float32")

    # ---- A. SEMANTIC SEARCH (finds meaning, not exact characters) -------------
    sims = M @ q
    top = np.argsort(-sims)[: args.k]
    print(f"=== 語意搜尋 semantic search: 「{args.query}」 ===")
    for rank, i in enumerate(top, 1):
        doc, c = chunks[i]
        print(f"#{rank} [{doc.episode_id} @ {_fmt_ts(c.start_s)}] score={sims[i]:.3f}")
        print(f"     {c.text[:90].strip()}…\n")

    # ---- B. AUTOMATIC SIMILAR WORDS (no hand-typed synonyms) ------------------
    # Embed the corpus vocabulary and rank terms by similarity to the query — this
    # is how the system "tries similar words" on its own.
    import jieba

    counts = Counter(
        t for txt in texts for t in jieba.lcut(txt) if len(t.strip()) >= 2 and t.strip()
    )
    vocab = [w for w, n in counts.most_common(600)]
    V = np.array(emb.embed(vocab), dtype="float32")
    vsims = V @ q
    sim_top = np.argsort(-vsims)[:10]
    similar_terms = [vocab[i] for i in sim_top]
    print(f"=== 自動相似詞 auto similar words for 「{args.query}」 ===")
    print("  " + "  ".join(f"{vocab[i]}({vsims[i]:.2f})" for i in sim_top) + "\n")

    # ---- C. KEYWORD (BM25) — exact terms, expanded with the similar words ------
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([jieba.lcut(t) for t in texts])
    expanded = " ".join([args.query] + similar_terms[:5])
    scores = bm25.get_scores(jieba.lcut(expanded))
    ktop = np.argsort(-scores)[: args.k]
    print(f"=== 關鍵字搜尋 keyword/BM25 (查詢已自動擴充: {expanded}) ===")
    hit = False
    for rank, i in enumerate(ktop, 1):
        if scores[i] <= 0:
            continue
        hit = True
        doc, c = chunks[i]
        print(f"#{rank} [{doc.episode_id} @ {_fmt_ts(c.start_s)}] bm25={scores[i]:.2f}")
        print(f"     {c.text[:90].strip()}…\n")
    if not hit:
        print("  （語料中沒有精準關鍵字命中——這正是語意搜尋更強的地方）\n")


if __name__ == "__main__":
    main()
