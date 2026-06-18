# Gooaye RAG — System & Milestones (living doc)

> This is the **source of truth** for how the system works and where we are.
> Updated as milestones progress. Last updated: **2026-06-17** — M0–M3 built &
> mock-verified; first free local 30-episode corpus building now.

---

## 1. What we're building, in one paragraph

A question-answering system over the **Gooaye 股癌** podcast. You ask a question in
Traditional Chinese; the system finds the relevant moments across hundreds of episodes
and has **Claude** write a grounded, cited answer. The interesting part — the part we
iterate on — is the **RAG layer** (how we retrieve). Everything around it (the web UI,
the API) is deliberately fixed and minimal so we can swap retrieval strategies freely
and compare them fairly.

## 2. The system has two phases

> 📊 Visual: [`system-structure.svg`](system-structure.svg) (open in the preview panel).


**Phase ① — Build the library (offline, once, on your Mac).**
Turn 671 episodes of audio into a searchable knowledge base. Runs locally; the only
thing that ever costs money here is the one-time graph extraction.

```
RSS feed ─▶ download audio ─▶ Whisper ASR ─▶ transcripts ─┬─▶ embed ─▶ Vector DB (Qdrant)
 (671 eps)   (temp, deleted)   (local, FREE)  (~60 MB)     └─▶ Claude extract ─▶ Knowledge Graph
```

**Phase ② — Answer a question (online, every query).**
```
your question ─▶ [strategy: Agentic | Graph] ─▶ retrieve from library ─▶ Claude writes answer (cited)
```

### Where money goes (and doesn't)
| Step | Cost | Notes |
|---|---|---|
| RSS, transcripts, Vector DB (Qdrant) | **FREE** | all local |
| Whisper ASR | **FREE** | local `mlx-whisper` on your Mac (Apple Silicon) |
| Embeddings | **~free** | local BGE-M3 = $0; or Voyage ~$0.03 / 30 eps |
| Knowledge Graph extraction | **~$2 once** | Claude Haiku, only when building Graph RAG |
| Claude answering a question | **~$0.01–0.10 / question** | depends on model (Opus/Sonnet/Haiku) |

**Bottom line:** building the library can be **$0**. You only need a paid (Anthropic)
key to *ask questions* and to build the *graph*.

## 3. The RAG strategies (the whole point)

Three are implemented (see also `docs/EVALUATION.md` for how we compare them):
- **Agentic RAG** — Claude drives its own retrieval via tools (vector + keyword +
  metadata + context-expansion); multi-step, query rewriting. Uses the Vector DB.
- **Graph RAG** — entities/relations extracted into a knowledge graph at build time;
  query = link entities → traverse → synthesize. Best for relationship/aggregation.
- **Corrective RAG (CRAG)** — retrieve → a cheap grader judges relevance → if weak,
  rewrite the query + switch retriever and try again → then answer. Self-correcting,
  robust to weak first hits.

### (legacy notes — the original two)

Both sit behind one interface (`backend/app/rag/base.py`) and are picked from a
dropdown. Adding/iterating a strategy never touches the API or the frontend.

- **Agentic RAG** (`app/rag/agentic/`) — Claude is the driver. It's given tools
  (`vector_search`, `keyword_search`, `get_episode_metadata`, `expand_context`) and
  decides itself what to search, whether to rewrite the query, and when it has enough
  to answer. Good at multi-step questions. Uses the **Vector DB**.
- **Graph RAG** (`app/rag/graph/`) — at build time, Claude extracts entities
  (companies, tickers, people, concepts) and their relations into a **knowledge graph**.
  At query time we find the entities in your question, walk the graph to related
  entities, gather the evidence passages, and Claude synthesizes from that. Good at
  "how do X and Y connect / who's involved in Z" questions.

Both emit the **same trace** (latency, tokens, cost, tool calls, #contexts) so M4 can
score them head-to-head.

## 4. Milestones — what each means

| # | Name | What it delivers | Status |
|---|---|---|---|
| **M0** | App shell + contract | FastAPI (`/strategies`, `/chat` SSE) + Next.js chat + strategy dropdown + `EchoStrategy` stub proving the pluggable design | ✅ done, tested |
| **M1** | Ingestion | RSS → local Whisper ASR (audio discarded) → timestamped transcripts → CJK chunking → embed → Qdrant. Idempotent; episode-count capped to keep disk small | ✅ built & tested; **first 30-ep corpus building now** |
| **M2** | Agentic RAG | Claude tool-use loop with 4 retrieval tools, streaming + cost trace, iteration cap | ✅ built, mock-verified |
| **M3** | Graph RAG | LLM entity/relation extraction → networkx graph → traversal retrieval → synthesis | ✅ built, mock-verified |
| **M4** | Eval harness | Golden Q&A set + RAGAS (faithfulness / context-recall / answer-relevancy) + latency/cost/tool-calls, **Agentic vs Graph** scorecard | ⏳ next |
| **M5** | Scale & maintain | Backfill full catalog + incremental "new episode" cron | later |

"Mock-verified" = the logic is proven end-to-end with a scripted fake LLM (no network);
"live-verified" happens once a real corpus + Anthropic key are in place.

## 5. Tech choices (and why)
- **Local Whisper (`mlx-whisper`, large-v3-turbo)** — free, fast on Apple Silicon, good
  Mandarin; audio streamed then deleted (never stored).
- **BGE-M3 local embeddings** (default for the free build) — $0, strong on zh-Hant;
  Voyage/OpenAI are drop-in alternatives via `EMBED_PROVIDER`.
- **Qdrant embedded mode** — vector search with no Docker; `data/qdrant_local/`. Switch
  to a server for deployment (`QDRANT_MODE=server`).
- **networkx + JSON** graph — tiny footprint for the test; swap for Neo4j if needed.
- **Claude** for answers (default Opus 4.8; `ANSWER_MODEL` to drop to Sonnet/Haiku for
  cost) and graph extraction (Haiku).

## 6. Current status / what's next
- ✅ M0–M3 code complete, 15/15 tests passing.
- ✅ ASR proven on real audio (EP671: 50 min → ~77 s, $0).
- ⏸️ **Corpus build paused at 14 episodes** (EP671→EP658), clean large-v3-turbo. Resume
  anytime: rerun the pipeline (it skips done episodes). 14 eps is enough to start testing.
- 📦 **Measured footprint:** ~248 KB/transcript, ~0.44 MB/ep in Qdrant → full 671-episode
  catalog ≈ **0.45 GB** of data. Audio is never stored. One-time models (~4 GB) shared.
- ✅ **Turnkey example ready:** paste key into `.env` → `python scripts/run_example.py`
  builds the graph (first run, ~$0.6) and answers a question through **both** strategies
  with a side-by-side trace comparison.
- 🔜 Then: **M4 eval** (golden Q&A + RAGAS) to score Agentic vs Graph rigorously.

## 6b. Known issues / decisions
- **Transcripts come out in Simplified Chinese** — Whisper large-v3-turbo defaults to
  Simplified, even for this Traditional-Chinese (Taiwan) podcast. Semantic search (BGE-M3) bridges the
  scripts fine, but exact keyword search + readability want Traditional. **Fix:** OpenCC
  s2t normalization in `app/ingestion/normalize.py` (free, local, idempotent — can be
  re-applied to existing transcripts without re-ASR).
- **Mode decision (2026-06-17):** running **retrieval-only ($0)** for now — no Claude
  answer generation. `scripts/demo_search.py` is the free interface (semantic + auto
  similar-words + keyword). Answer generation (Claude or a local LLM) is deferred.
  Note: retrieval *quality* can still be evaluated for free (recall/precision) without
  an LLM; full RAGAS (M4) needs a generator.

## 6c. Hardening done (production-readiness)
- **#1 Index-version guard** — the vector index now stores a *manifest*
  (embed_provider, embed_model, dim, chunk params). Indexing writes it; querying
  (`RetrievalToolbox`) verifies it and raises `IndexMismatchError` on any mismatch.
  Catches the silent trap where voyage-3 and BGE-M3 are *both 1024-d* — a dim-only
  check would miss it. See `app/retrieval/vector_store.py`.
- **#6 API security** — `/chat` + `/strategies` now support bearer/`X-API-Key` auth
  (`API_AUTH_TOKEN`; empty = open dev mode, logged as a warning), a per-IP rate limit
  (`RATE_LIMIT_PER_MIN`), and file-based secrets (`*_FILE` env, Docker/K8s convention).
  See `app/main.py`, `app/config.py`.
- Still deferred for production (from the gap analysis): server-mode Qdrant +
  external graph/BM25 for concurrency, observability stack, retries/fallbacks/caching,
  larger governed eval set + CI gate, reranker + better entity resolution.

## 6d. Free deployment (in progress)
- Target: **Hugging Face Spaces (free, 16 GB) + Gradio UI**, answers via **free Gemini**.
- Only Phase-2 serving is deployed; the prebuilt Qdrant index + transcripts ship with
  the Space (no ASR on the server). Query embeddings = local BGE-M3 (needs the RAM).
- **Provider seam** now pluggable: `LLM_PROVIDER=anthropic|gemini`, `get_llm()` in
  `app/rag/agentic/llm.py`. `GeminiLLM` does MANUAL function calling and translates our
  Claude-style block format ↔ Gemini Content/Part, so all 3 strategies (incl. agentic
  tool-loop) work on Gemini. Unit-verified (26 tests).
- Remaining: Gradio app (`space/app.py`), Space `requirements.txt`/`README.md`, ship the
  index, set `GEMINI_API_KEY` secret + path env, push to the Space.

## 7. Update log
- **2026-06-17** — M0–M3 built & mock-verified. Added free local path (mlx-whisper +
  BGE-M3, embedded Qdrant) so the corpus build needs no keys/money. Started first
  30-episode build (paused at 14). This doc created as the living system reference. Added
  `system-structure.svg` plot and `scripts/demo_search.py` (local retrieval inspector).
- **2026-06-18** — Hardening: added **index-version guard** (#1) and **API
  auth + rate limit + file secrets** (#6); 23 tests green; reindexed to stamp the
  manifest. Added the 3-RAG structure diagram (`docs/rag-strategies.svg`) and a top-level
  `README.md`.
- **2026-06-17 (latest)** — Added **Corrective RAG** (3rd strategy; 17 tests green).
  Wrote `docs/EVALUATION.md` (experiment design: golden set, retrieval vs generation
  tiers, metrics, fairness/honesty guards) + seed `eval/golden.jsonl` +
  `scripts/eval_retrieval.py`. Reindexed Qdrant to Traditional. Ran the **free Tier-1
  retrieval eval**: hybrid best Recall@8 (0.77), vector best MRR (0.80), keyword wins
  `relationship` (1.00). Tier-2 (RAGAS answer quality) still pending a key.
- **2026-06-17 (later)** — Chose retrieval-only ($0) mode. Found transcripts were
  Simplified Chinese; added OpenCC s2twp to the normalizer + `scripts/to_traditional.py`,
  converted the 14 transcripts to Traditional in place. Retrieval verified on clean data
  (the "US stocks" query ~0.55; the show-name-origin query found at 0.48). Known minor
  ASR artifacts on proper nouns (the show's own name and the host's name get mis-heard)
  — fixable later with a Whisper `initial_prompt`.
