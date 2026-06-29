"""Demo the query-suggestion / next-step prediction — FREE (no LLM by default).

    python scripts/suggest_demo.py "輝達"        # vague → top-3 suggestions
    python scripts/suggest_demo.py "股癌怎麼看美股的修正？"   # clear → no suggestions
    python scripts/suggest_demo.py "輝達" --llm   # use the LLM to phrase nicer questions
"""

from __future__ import annotations

import argparse
import asyncio

from app.config import get_settings


async def main_async(query: str, use_llm: bool) -> None:
    s = get_settings()
    s.embed_provider = "local"
    s.suggest_use_llm = use_llm
    from app.rag.suggest import QuerySuggester

    res = await QuerySuggester(s).suggest(query)
    print(f'\n查詢：「{query}」  →  ambiguous={res.ambiguous} ({res.reason})')
    if res.ambiguous:
        print("\n🤔 你的問題有點籠統，你想問的是？")
        for i, sg in enumerate(res.suggestions, 1):
            eps = ",".join(sg.episodes)
            print(f"  {i}. {sg.question}" + (f"   [{eps}]" if eps else ""))
    else:
        print("（問題夠清楚，直接回答即可）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--llm", action="store_true", help="use the LLM to phrase suggestions")
    args = ap.parse_args()
    asyncio.run(main_async(args.query, args.llm))


if __name__ == "__main__":
    main()
