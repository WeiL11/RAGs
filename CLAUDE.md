# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Living system doc:** [`docs/SYSTEM.md`](docs/SYSTEM.md) explains the phases,
> milestones M0–M5, the two RAG strategies, and costs — keep it updated as work lands.

## What this is

A RAG question-answering system over the **Gooaye 股癌** podcast (Traditional Chinese
investing podcast). The defining constraint: **the RAG layer is the product and changes
constantly; the app shell (FastAPI + Next.js) is intentionally fixed.** Multiple RAG
strategies live behind one interface and are swapped by a dropdown. Two are planned —
**Agentic RAG** and **Graph RAG** — to be compared on a RAGAS eval harness. `echo` is a
stub strategy that exists only to prove the pipeline.

## Commands

Backend (from `backend/`, after `source .venv/bin/activate`):
```bash
pip install -e ".[dev]"                 # core + test deps (M0 runs with just this)
pip install -e ".[ingestion,asr-mlx]"   # M1: ingestion + local Whisper (Apple Silicon)
uvicorn app.main:app --reload --port 8000
pytest                                   # all tests
pytest tests/test_contract.py::test_chat_streams_sse   # a single test
ruff check . && ruff format .
python -m app.ingestion.pipeline --limit 5             # ingest 5 newest episodes
python -m app.ingestion.pipeline --transcribe-only     # ASR only, skip vector index
python -m app.ingestion.pipeline --reindex             # re-embed existing transcripts
```

Frontend (from `frontend/`):
```bash
npm install && npm run dev    # http://localhost:3000
npm run build                 # also type-checks
```

## Architecture — the parts that span files

**The contract is everything.** `backend/app/rag/base.py` defines `BaseRAGStrategy`
(abstract), `RetrievedContext`, `StreamEvent`, and `Message`. The FastAPI shell
(`app/main.py`) and the frontend depend ONLY on these. Adding/iterating a strategy must
never require touching `main.py` or the frontend — if it does, the abstraction is wrong.

**Strategy lifecycle:** implement `BaseRAGStrategy` → register it in
`app/rag/registry.py::build_default_registry` (one `elif`) → add its name to
`ENABLED_STRATEGIES`. The first enabled strategy is the UI default. Strategies are
**lazy-imported** in the registry so the app runs without their (heavy) deps installed.
`GET /strategies` drives the frontend dropdown; `POST /chat` streams `StreamEvent`s as
SSE (`{type: "contexts"|"token"|"done"|"error"}`). The frontend parses these in
`frontend/app/page.tsx`.

**Offline ingestion vs. online serving are separate.** `app/ingestion/` is a one-time
batch job (RSS → ASR → transcript → chunk → embed → Qdrant) run locally; the deployed
backend only does query → embed → retrieve → Claude. How transcripts are produced (ASR
engine, episode count) has **zero impact on deployability**. Audio is streamed to a temp
file, transcribed, and **deleted** — never persisted (see `app/ingestion/asr.py`).

**Pipeline idempotency** (`app/ingestion/pipeline.py`): an episode whose transcript JSON
exists is reused (ASR skipped); an episode already in Qdrant is skipped at index time.
`--reindex` re-embeds without re-transcribing; `--force` redoes ASR. `episode_window` /
`--limit` caps how many newest episodes are processed (keeps device footprint small).
The canonical source of truth is one `TranscriptDoc` JSON per episode under
`data/transcripts/`; chunks are derived at index time, not stored separately.

**Everything pluggable goes through a factory + config**, so iteration is config not code:
- ASR engine: `asr.py::transcribe_file` dispatches on `asr_provider` (`mlx` default /
  `faster-whisper` / `openai`). Engine libs imported lazily.
- Embeddings: `retrieval/embedder.py::get_embedder` returns Voyage (default, 1024-d) or
  OpenAI (3072-d) behind the `Embedder` ABC. `input_type` ("document"/"query") matters
  for Voyage retrieval quality.
- Vector store: `retrieval/vector_store.py`. `qdrant_mode="local"` uses **embedded
  Qdrant** (on-disk under `data/qdrant_local`, or `:memory:` for tests) — no Docker
  needed; `"server"` connects to `qdrant_url` for deployment. Payload schema carries
  everything to build a `RetrievedContext` (text, episode_id, ep_number, publish_date,
  start_s/end_s). Uses `query_points` (not the deprecated `search`).

All settings live in `app/config.py` (pydantic-settings, reads repo-root `.env`).

## Conventions specific to this codebase

- **Chinese text has no word spaces** — chunk by *character count* on CJK sentence
  boundaries (`。！？…；`), not whitespace. See `app/ingestion/chunker.py`. Chunks carry
  the timestamp range of the segments they span so the UI can deep-link to audio.
- Gooaye episode **titles are emoji-only** (e.g. `EP671 | 🌼`); all topical signal comes
  from transcript content, not titles. Episode numbers are parsed from the title.
- The RSS feed is the SoundOn feed in `config.py::rss_url` (~671 episodes). `feedparser`
  unwraps the CDATA titles.
- Every strategy's `answer` must emit a final `"done"` event whose `trace` carries
  latency/tokens/cost/tool-calls — the eval harness (M4) consumes this shape.
- M0/M1 require no API keys to *run the code*; keys are only needed at ingest time
  (embeddings) and query time (Claude). The `echo` strategy needs nothing.

## Milestones

M0 ✅ app shell + pluggable contract + `EchoStrategy`. M1 ✅ ingestion (RSS→ASR→
transcripts→Qdrant). M2 Agentic RAG (Claude tool-use). M3 Graph RAG. M4 RAGAS eval +
strategy comparison. M5 scale to full catalog + incremental cron. Plan detail lives in
`README.md` and the design notes.
