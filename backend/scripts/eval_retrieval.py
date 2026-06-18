"""FREE retrieval-quality evaluation — no LLM, no API key.

Compares the retrievers the strategies rely on — vector (BGE-M3), keyword (BM25),
and hybrid (reciprocal-rank fusion) — against the golden set's gold_episodes.
Reports Hit@k / Recall@k / MRR overall and per category. This is Tier 1 of the
experiment in docs/EVALUATION.md.

    python scripts/eval_retrieval.py            # k=8
    python scripts/eval_retrieval.py -k 5
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from app.config import get_settings
from app.eval.dataset import load_golden


def _unique_episodes(contexts) -> list[str]:
    seen, order = set(), []
    for c in contexts:
        if c.episode_id not in seen:
            seen.add(c.episode_id)
            order.append(c.episode_id)
    return order


def _metrics(retrieved: list[str], gold: list[str]) -> dict[str, float]:
    gold_set = set(gold)
    inter = [e for e in retrieved if e in gold_set]
    hit = 1.0 if inter else 0.0
    recall = len(set(inter)) / len(gold_set) if gold_set else 0.0
    mrr = 0.0
    for rank, e in enumerate(retrieved, 1):
        if e in gold_set:
            mrr = 1.0 / rank
            break
    return {"hit": hit, "recall": recall, "mrr": mrr}


def _rrf(*ranked_lists: list[str], k: int = 60) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, ep in enumerate(lst, 1):
            score[ep] += 1.0 / (k + rank)
    return [ep for ep, _ in sorted(score.items(), key=lambda x: -x[1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    s = get_settings()
    s.embed_provider = "local"
    questions = load_golden()
    scored = [q for q in questions if q.category != "negative"]
    negatives = [q for q in questions if q.category == "negative"]
    print(f"golden: {len(questions)} questions ({len(scored)} scored, {len(negatives)} negative)\n")

    from app.retrieval.embedder import get_embedder
    from app.retrieval.keyword import KeywordIndex
    from app.retrieval.vector_store import VectorStore

    emb = get_embedder(s)
    store = VectorStore(s)
    kw = KeywordIndex(s).build()
    print(f"retrievers ready (embed={emb.model}, bm25 chunks={kw.size})\n")

    def vector(q):  # noqa: ANN001
        return _unique_episodes(store.search(emb.embed_query(q), k=args.k))

    def keyword(q):  # noqa: ANN001
        return _unique_episodes(kw.search(q, k=args.k))

    def hybrid(q):  # noqa: ANN001
        return _rrf(vector(q), keyword(q))[: args.k]

    retrievers = {"vector": vector, "keyword": keyword, "hybrid": hybrid}
    agg: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in retrievers
    }
    by_cat: dict[tuple[str, str], list[float]] = defaultdict(list)

    for q in scored:
        for name, fn in retrievers.items():
            m = _metrics(fn(q.question), q.gold_episodes)
            for key, val in m.items():
                agg[name][key].append(val)
            by_cat[(name, q.category)].append(m["recall"])

    print(f"=== Retrieval scorecard (k={args.k}, {len(scored)} questions) ===")
    print(f"{'retriever':10} {'Hit@k':>7} {'Recall@k':>9} {'MRR':>7}")
    for name in retrievers:
        a = agg[name]
        print(
            f"{name:10} {_mean(a['hit']):7.2f} {_mean(a['recall']):9.2f} {_mean(a['mrr']):7.2f}"
        )

    cats = sorted({q.category for q in scored})
    print(f"\n=== Recall@k by category ===")
    print(f"{'retriever':10} " + " ".join(f"{c:>12}" for c in cats))
    for name in retrievers:
        row = " ".join(f"{_mean(by_cat[(name, c)]):12.2f}" for c in cats)
        print(f"{name:10} {row}")

    print(f"\nNegative questions ({[q.id for q in negatives]}): retrieval can't score these — "
          "they test hallucination in the generation tier (a good answer admits it has none).")
    print("\nNote: this is Tier-1 (retrieval) only. Graph retrieval + answer quality (RAGAS) "
          "need the graph built + an API key — see docs/EVALUATION.md.")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


if __name__ == "__main__":
    main()
