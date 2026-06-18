# Gooaye (股癌) Podcast RAG

A question-answering system over the **Gooaye 股癌** podcast (Traditional-Chinese
investing podcast, ~671 episodes). Ask a question; the system finds the relevant
moments across episodes and answers with citations.

It's built as a **RAG research harness**: the retrieval layer is the product, so
multiple RAG strategies sit behind one interface and are swapped from a dropdown.
The app shell around them stays fixed. **The whole library can be built for $0** — only
the optional answer-writing step uses a paid LLM.

📊 Diagrams: [system structure](docs/system-structure.svg) · [the 3 RAG strategies](docs/rag-strategies.svg)
📖 Deep dives: [docs/SYSTEM.md](docs/SYSTEM.md) (architecture & milestones) ·
[docs/EVALUATION.md](docs/EVALUATION.md) (how we compare strategies)

## How it works (two phases)

**① Build the library — once, on your Mac, free.**
RSS → download audio → **Whisper** transcription (local) → Simplified→Traditional
(OpenCC) → CJK-aware **chunking** → **BGE-M3 embeddings** (local) → **Qdrant** vector
index, and optionally an LLM-extracted **knowledge graph**. Audio is streamed and
deleted — never stored.

**② Answer a question — per query.**
question → pick a strategy → retrieve from the library → (optionally) an LLM writes a
cited answer. Retrieval is free and local; only the final generation costs money.

> Two kinds of model, and only one is the expensive one: the **embedding model**
> (BGE-M3, ~0.6B) runs free on your Mac for *search*; the **generative LLM** (Claude)
> only writes the *answer* — and is optional (retrieval-only, or a local LLM, both $0).

## The three RAG strategies

| Strategy | How it retrieves | Best at | Per-query LLM cost |
|---|---|---|---|
| **Agentic** | Claude drives a tool loop (vector + keyword + metadata + expand) | multi-step, opinion lookups | higher (multi-call) |
| **Graph** | link entities → traverse knowledge graph → synthesize | relationships, cross-episode aggregation | lowest (1 call) |
| **Corrective (CRAG)** | retrieve → grade relevance → rewrite & re-retrieve if weak | robustness, weak first hits | medium (grader + answer) |

All three implement the same contract ([`backend/app/rag/base.py`](backend/app/rag/base.py))
and register in [`registry.py`](backend/app/rag/registry.py) — adding one never touches
the API or the frontend. See [the structure diagram](docs/rag-strategies.svg).

## Quickstart

**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ingestion,rag,asr-mlx,embed-local]"
```

**Build a small free corpus** (local Whisper + BGE-M3, no keys):
```bash
python -m app.ingestion.pipeline --limit 14   # transcribe + index 14 newest episodes
```

**Search it — free, no LLM:**
```bash
python scripts/demo_search.py "美股"           # "US stocks" — semantic + similar-words + keyword
python scripts/eval_retrieval.py               # retrieval-quality scorecard
```

**Get written answers** (needs an Anthropic key — paste into `.env`, ~$0.10/question):
```bash
python scripts/run_example.py                  # asks one question via all three strategies
```

**Run the app** (FastAPI + Next.js chat with a strategy dropdown):
```bash
uvicorn app.main:app --reload --port 8000      # backend
cd ../frontend && npm install && npm run dev    # http://localhost:3000
```

**Tests:** `cd backend && pytest` (23 passing).

## What's free vs paid
- **Free, local:** transcription, embeddings, vector + keyword search, Traditional
  conversion, retrieval-quality eval.
- **Paid (optional):** Claude writing answers (~$0.01–0.10/question) and building the
  knowledge graph (~$1 one-time). Swap to a local LLM to make these free too.

## Project layout
```
gooaye-rag/
  backend/app/
    rag/         base contract + registry + agentic/ graph/ corrective/ echo/
    retrieval/   embedder · vector_store (Qdrant + index-version guard) · keyword · graph_store
    ingestion/   rss · asr · normalize (OpenCC) · chunker · pipeline · build_graph
    eval/        golden-set loader (+ scripts/eval_retrieval.py)
    main.py      FastAPI: /chat, /strategies (auth + rate limit), /health
  frontend/      minimal Next.js chat + strategy dropdown
  eval/golden.jsonl   versioned Q&A set
  docs/          SYSTEM.md, EVALUATION.md, diagrams
  data/          transcripts / qdrant / graph   (gitignored; audio never stored)
```

## Security & robustness
- API auth (`API_AUTH_TOKEN`, bearer / `X-API-Key`) + per-IP rate limit; file-based
  secrets (`*_FILE`). Set a token before exposing on a network.
- **Index-version guard:** the vector index records its embedding model/dim/chunk
  params and refuses queries from a mismatched config (prevents silent corruption).
- Known production gaps (concurrency, observability, retries/caching, reranking) are
  tracked in [docs/SYSTEM.md](docs/SYSTEM.md) §6c.

## Status
M0–M3 complete + Corrective RAG; 3 strategies ready. 14-episode corpus built and
searchable (free). Next: expand the eval set and run the full RAGAS comparison (M4).
