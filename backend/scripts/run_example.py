"""One-command example: ask the same question through BOTH RAG strategies.

    cd backend && source .venv/bin/activate
    python scripts/run_example.py                      # default example question
    python scripts/run_example.py "股癌怎麼看輝達？"     # your own question
    python scripts/run_example.py --only agentic       # just one strategy

On first run it builds the knowledge graph (needed by Graph RAG) from the existing
transcripts using Claude Haiku (~$0.6 for the 14-episode corpus). Requires
ANTHROPIC_API_KEY in ../.env. Retrieval/embeddings stay local and free.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.config import get_settings

EXAMPLE_Q = "股癌最近這幾集聊到哪些投資主題或市場看法？請舉具體例子並標註集數。"


def _ts(s: float | None) -> str:
    s = s or 0.0
    return f"{int(s // 60)}:{int(s % 60):02d}"


async def run_strategy(label: str, strat, query: str) -> dict | None:
    print(f"\n{'=' * 72}\n▶ {label}\n{'=' * 72}")
    contexts, trace = [], {}
    try:
        async for ev in strat.answer(query):
            if ev.type == "token":
                print(ev.delta, end="", flush=True)
            elif ev.type == "contexts":
                contexts = ev.contexts
            elif ev.type == "done":
                trace = ev.trace
            elif ev.type == "error":
                print(f"\n[ERROR] {ev.delta}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAILED] {type(exc).__name__}: {exc}")
        return None

    print("\n\n— 來源 sources —")
    for c in contexts[:5]:
        print(f"  [{c.episode_id} @ {_ts(c.start_s)}] {c.source} {c.score:.3f} :: {c.text[:46].strip()}…")
    print(
        f"— trace — model={trace.get('model')} latency={trace.get('latency_ms')}ms "
        f"cost=${trace.get('cost_usd')} tool_calls={trace.get('tool_calls')} "
        f"contexts={trace.get('n_contexts')}"
    )
    return trace


async def main_async(query: str, only: list[str] | None) -> int:
    s = get_settings()
    if not s.anthropic_api_key:
        print(
            "⚠️  尚未設定 ANTHROPIC_API_KEY。\n"
            "    請編輯 gooaye-rag/.env，貼上你的金鑰：\n"
            "        ANTHROPIC_API_KEY=sk-ant-...\n"
            "    然後重新執行此指令。"
        )
        return 1
    if s.embed_provider != "local":
        print(
            f"⚠️  EMBED_PROVIDER={s.embed_provider!r}，但向量索引是用 local BGE-M3 建立的。"
            "請在 .env 設 EMBED_PROVIDER=local。"
        )
        return 1

    want = only or ["agentic", "graph", "corrective"]
    print(f"問題：{query}\n策略：{', '.join(want)}　語料：{s.transcripts_dir}")

    # Build the graph once if Graph RAG is requested and it doesn't exist yet.
    if "graph" in want and not Path(s.graph_path).exists():
        print("\n第一次執行 Graph RAG：建立知識圖譜（Claude Haiku 抽取，約 $0.6）…")
        from app.ingestion.build_graph import build_graph

        build_graph(s)

    from app.rag.agentic.strategy import AgenticRAGStrategy
    from app.rag.graph.strategy import GraphRAGStrategy

    traces: dict[str, dict | None] = {}
    if "agentic" in want:
        traces["agentic"] = await run_strategy("Agentic RAG", AgenticRAGStrategy(s), query)
    if "graph" in want:
        traces["graph"] = await run_strategy("Graph RAG", GraphRAGStrategy(s), query)

    valid = {k: t for k, t in traces.items() if t}
    if len(valid) > 1:
        print(f"\n{'=' * 72}\n比較 comparison\n{'=' * 72}")
        for k, t in valid.items():
            print(
                f"  {k:8} latency={t.get('latency_ms')}ms  cost=${t.get('cost_usd')}  "
                f"tool_calls={t.get('tool_calls')}  contexts={t.get('n_contexts')}"
            )
        print("\n→ 這就是 M4 評估要量化比較的東西（再加上 RAGAS 正確性指標）。")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an example through both RAG strategies")
    ap.add_argument("query", nargs="?", default=EXAMPLE_Q)
    ap.add_argument("--only", choices=["agentic", "graph"], default=None)
    args = ap.parse_args()
    only = [args.only] if args.only else None
    sys.exit(asyncio.run(main_async(args.query, only)))


if __name__ == "__main__":
    main()
