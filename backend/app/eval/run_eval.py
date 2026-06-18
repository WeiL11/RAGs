"""M4 — compare the RAG strategies end-to-end on the golden set.

For each (strategy, question): run the strategy, capture the answer + retrieved
contexts + trace, compute free retrieval recall vs gold_episodes, and score the
answer with the LLM judge. Aggregates a scorecard (overall + per category).

⚠️ Cost/quota: a full run makes many LLM calls — roughly
    questions × Σ_strategy(generation_calls + 1 judge_call)
graph≈1, corrective≈2, agentic≈2–4 generation calls each, plus 1 judge call.
On Gemini's free tier use `--sample` to stay within the daily quota, e.g. `--sample 6`.

    python -m app.eval.run_eval --sample 6
    python -m app.eval.run_eval --strategies graph,corrective --sample 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config import DATA_DIR, get_settings
from app.eval.dataset import EvalQuestion, load_golden


async def _run_one(strat, q: EvalQuestion) -> dict[str, Any]:
    answer, contexts, trace, error = "", [], {}, None
    try:
        async for ev in strat.answer(q.question):
            if ev.type == "token":
                answer += ev.delta
            elif ev.type == "contexts":
                contexts = ev.contexts
            elif ev.type == "done":
                trace = ev.trace
            elif ev.type == "error":
                error = ev.delta
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {"answer": answer, "contexts": contexts, "trace": trace, "error": error}


def retrieval_recall(contexts, gold: list[str]) -> float | None:
    if not gold:  # negative question — retrieval recall is N/A
        return None
    got = {c.episode_id for c in contexts}
    return len(got & set(gold)) / len(set(gold))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean of each metric per strategy (retrieval_recall ignores None/negatives)."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["strategy"]].append(r)

    def _mean(xs: list[float]) -> float:
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 3) if xs else 0.0

    out: dict[str, dict[str, float]] = {}
    for name, rs in by.items():
        out[name] = {
            "faithfulness": _mean([r["faithfulness"] for r in rs]),
            "correctness": _mean([r["correctness"] for r in rs]),
            "relevance": _mean([r["relevance"] for r in rs]),
            "retrieval_recall": _mean([r["retrieval_recall"] for r in rs]),
            "latency_ms": _mean([r.get("latency_ms") for r in rs]),
            "errors": float(sum(1 for r in rs if r.get("error"))),
        }
    return out


async def main_async(args: argparse.Namespace) -> int:
    from app.eval.judge import judge_answer
    from app.rag.registry import build_default_registry

    s = get_settings()
    registry = build_default_registry(s)
    names = [n.strip() for n in args.strategies.split(",")] if args.strategies else registry.names()
    names = [n for n in names if n in registry.names() and n != "echo"]

    questions = load_golden()
    if args.sample:
        questions = questions[: args.sample]

    print(f"eval: {len(questions)} questions × {len(names)} strategies = "
          f"{len(questions) * len(names)} runs ({names})\n")

    rows: list[dict[str, Any]] = []
    for q in questions:
        for name in names:
            res = await _run_one(registry.get(name), q)
            rec = retrieval_recall(res["contexts"], q.gold_episodes)
            scores = (
                await judge_answer(s, q.question, q.reference, res["answer"], res["contexts"])
                if res["answer"] and not res["error"]
                else {"faithfulness": 0.0, "correctness": 0.0, "relevance": 0.0}
            )
            row = {
                "id": q.id, "category": q.category, "strategy": name,
                "retrieval_recall": rec, **scores,
                "latency_ms": (res["trace"] or {}).get("latency_ms"),
                "tool_calls": (res["trace"] or {}).get("tool_calls"),
                "error": res["error"], "answer": res["answer"][:500],
            }
            rows.append(row)
            print(f"  {q.id:4} {name:11} faith={scores['faithfulness']:.2f} "
                  f"corr={scores['correctness']:.2f} rel={scores['relevance']:.2f} "
                  f"recall={rec if rec is None else round(rec,2)}"
                  f"{'  ERR:'+res['error'] if res['error'] else ''}")

    # --- scorecard ---
    card = aggregate(rows)
    print(f"\n=== Scorecard ({len(questions)} questions) ===")
    print(f"{'strategy':12} {'faith':>6} {'correct':>8} {'relev':>6} {'recall':>7} {'lat(ms)':>8} {'err':>4}")
    for name, m in card.items():
        print(f"{name:12} {m['faithfulness']:6.2f} {m['correctness']:8.2f} {m['relevance']:6.2f} "
              f"{m['retrieval_recall']:7.2f} {m['latency_ms']:8.0f} {int(m['errors']):4}")

    out_dir = Path(DATA_DIR) / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval_{args.tag or 'latest'}.json"
    out.write_text(json.dumps({"scorecard": card, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nsaved → {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M4 — compare RAG strategies on the golden set")
    p.add_argument("--sample", type=int, default=0, help="limit to first N questions (free-tier)")
    p.add_argument("--strategies", default="", help="comma list, e.g. graph,corrective")
    p.add_argument("--tag", default="", help="filename tag for the saved results")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
