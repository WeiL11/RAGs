# Gooaye (股癌) Podcast RAG

A RAG system that finds the relevant moments across podcast episodes and answers with
citations — focused on **RECENT** episodes, so users can recap recent episodes instead
of re-listening to whole episodes.

It's built as a **RAG research harness** over the **Gooaye 股癌** podcast (a
Traditional-Chinese investing podcast): the retrieval layer is the product, so multiple
RAG strategies sit behind one interface and are swapped from a dropdown. The app shell
around them stays fixed. **The whole library can be built for $0** — only the optional
answer-writing step uses a paid LLM.

## Corpus coverage

- **Currently indexed: EP658–EP671 (14 episodes)** — the most recent episodes.
- **Auto-updates** to the newest episodes: an `update_corpus` workflow
  (`scripts/update_corpus.py`) pulls the latest from the RSS feed, transcribes and
  indexes them, and trims the window so the corpus stays focused on recent episodes.
- The app and CLI report the covered recent-episode range via a `corpus_range()` status
  helper, so it's always clear which episodes an answer can draw from.

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
> (BGE-M3, ~0.6B) runs free on your Mac for *search*; the **generative LLM** only writes
> the *answer* — and is optional (retrieval-only, or a local LLM, both $0).

## The three RAG strategies

All three implement the same contract ([`backend/app/rag/base.py`](backend/app/rag/base.py))
and register in [`registry.py`](backend/app/rag/registry.py) — adding one never touches
the API or the frontend. See [the rendered diagram](docs/rag-strategies.svg). Below,
🧠 marks an LLM call.

### Agentic RAG — LLM-driven tool loop

```
              ┌─────────────────────── loop ───────────────────────┐
              ▼                                                     │
Query → 🧠 plan / pick tool → search (vector + keyword) → 🧠 enough? ┤
                                                                    │ no
                                                              yes   │
                                                                ▼   ┘
                                                        🧠 cited answer
```

Claude drives the loop, deciding which tool to call and when it has enough evidence.
**Best for** multi-step / open-ended questions. **Higher cost** (2–6 LLM calls).

### Graph RAG — knowledge-graph traversal

```
Query → find entities → traverse graph (neighbors) → gather passages + relations
                                                              │
                                                              ▼
                                                  🧠 synthesize answer
```

Links the query to entities, walks the knowledge graph, then synthesizes once.
**Best for** relationships / cross-episode aggregation. **Cheapest** (1 LLM call).

### Corrective RAG / CRAG — self-correcting

```
                  ┌──────────── loop if weak ────────────┐
                  ▼                                       │
Query → retrieve → 🧠 grade relevance → weak? → rewrite + re-retrieve
                          │
                          │ kept passages
                          ▼
                  🧠 answer from kept passages
```

Grades each retrieval; if the hits are weak it rewrites the query and re-retrieves before
answering. **Best for** weak / ambiguous first hits. **Medium cost** (grader + answer).

### Comparison

| Strategy | How it retrieves | Best at | Per-query LLM cost |
|---|---|---|---|
| **Agentic** | LLM drives a tool loop (vector + keyword), looping until satisfied | multi-step, open-ended | higher (2–6 calls) |
| **Graph** | link entities → traverse knowledge graph → synthesize | relationships, cross-episode aggregation | lowest (1 call) |
| **Corrective (CRAG)** | retrieve → grade → rewrite & re-retrieve if weak | weak / ambiguous first hits | medium (grader + answer) |

## Performance / results

**FREE Tier-1 retrieval scorecard** — BGE-M3 embeddings, on the 30-question golden set
([`eval/golden.yaml`](eval/golden.yaml)), k=8. No LLM, no API key.

| Retriever | Hit@8 | Recall@8 | MRR |
|---|---|---|---|
| vector | 0.96 | 0.72 | 0.72 |
| keyword | 0.89 | 0.67 | 0.71 |
| **hybrid** | **0.93** | **0.82** | **0.76** |

**Hybrid wins overall and in every category.** By-category Recall@8 for hybrid:

| Category | aggregation | lookup | multi_hop | opinion | relationship |
|---|---|---|---|---|---|
| Recall@8 | 1.00 | 0.83 | 0.89 | 0.71 | 0.78 |

An answer-quality eval harness also exists (`python -m app.eval.run_eval`), but full
answer-quality numbers are pending LLM budget.

**Reproduce the retrieval scorecard for free:**
```bash
cd backend && python scripts/eval_retrieval.py
```

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

**Keep the corpus on the most recent episodes:**
```bash
python scripts/update_corpus.py               # fetch + index new episodes, trim the window
```

**Search it — free, no LLM:**
```bash
python scripts/demo_search.py "美股"           # "US stocks" — semantic + similar-words + keyword
python scripts/eval_retrieval.py               # retrieval-quality scorecard
python scripts/suggest_demo.py "輝達"          # query suggestion / next-step prediction
```

**Get written answers** (needs an LLM key — paste into `.env`):
```bash
python scripts/run_example.py                  # asks one question via all three strategies
python -m app.eval.run_eval                    # answer-quality comparison harness
```

There are **3 LLM providers** — `anthropic` (Claude), `gemini`, and `groq` — selected via
`LLM_PROVIDER` in `.env`. The retrieval path (BGE-M3 + Qdrant) is fully local and free
regardless of provider.

**Run the app** (FastAPI + Next.js chat with a strategy dropdown):
```bash
uvicorn app.main:app --reload --port 8000      # backend
cd ../frontend && npm install && npm run dev    # http://localhost:3000
```

**Tests:** `cd backend && pytest`.

## What's free vs paid
- **Free, local:** transcription, embeddings, vector + keyword + hybrid search,
  Traditional conversion, query suggestion, retrieval-quality eval. Groq and Gemini both
  offer generous free tiers for the answer step.
- **Paid (optional):** Claude writing answers (~$0.01–0.10/question) and building the
  knowledge graph (~$0.6 one-time for the 14-episode corpus). Swap to a local LLM or a
  free-tier provider to make these free too.

## Project layout
```
gooaye-rag/
  backend/app/
    rag/         base contract + registry + agentic/ graph/ corrective/ echo/ + suggest.py
    retrieval/   embedder · vector_store (Qdrant + index-version guard) · keyword · graph_store
    ingestion/   rss · asr · normalize (OpenCC) · chunker · pipeline · build_graph
    eval/        golden-set loader · run_eval (answer quality) · judge
    main.py      FastAPI: /chat, /strategies (auth + rate limit), /health
  backend/scripts/  demo_search · eval_retrieval · run_example · suggest_demo · update_corpus
  frontend/      minimal Next.js chat + strategy dropdown
  eval/golden.yaml    versioned 30-question Q&A set
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

## Test design

Tests are layered by *what each layer is afraid of*, so most of the suite runs offline,
for free, in seconds (`cd backend && pytest` → 46 tests in ~3s).

1. **Contract tests** (`test_contract.py`) — *fear: the abstraction leaks.* Using only the
   `echo` stub, assert `/chat` streams SSE and `/strategies` lists strategies. This is the
   litmus test of the whole design: it proves adding a strategy never touches the app shell.
2. **Unit tests + a Fake LLM** — *fear: the logic is wrong.* Every strategy takes an
   injected `FakeLLM` with pre-recorded responses, so the Agentic tool-loop, the Corrective
   grade→rewrite→re-retrieve flow, the CJK chunker (sentence boundaries + ad-stripping), and
   the suggester (ambiguity detection) are all tested with **no network, no key, no cost**.
3. **Evaluation harness** — *fear: it runs but answers badly.* Split in two: a **free
   retrieval scorecard** (Recall@k / MRR, no LLM — see the table above) and an
   **answer-quality harness** (`app.eval.run_eval`, LLM-as-judge for faithfulness /
   correctness / relevance). The golden set is `eval/golden.yaml` (30 questions),
   human-verified. This layer is what produced the "Corrective is best" conclusion
   (corrective 0.93/0.93/0.97 with 0 errors vs. agentic 0.50/0.50/0.50 with 2 errors —
   Groq's Llama function-calling is unreliable, which the harness caught, not intuition).

## Engineering log: challenges & solutions

A by-category record of what went wrong and how it was fixed — kept as a portfolio of the
real problems behind the architecture decisions above.

**A. Data-source reality.**
- *Third-party transcripts were unreliable* — gooayetranscript.com is a JS-rendered SPA
  exposing only ~30 episodes with inconsistent slugs. → **Generate transcripts ourselves**
  via ASR from the RSS audio feed: complete, timestamped, legally cleaner, fully owned.
- *Disk-footprint fear* — the full catalog is 18–25 GB of audio. → **Stream audio,
  transcribe, delete immediately**; persist only transcript JSON (<500 MB) and gate
  ingestion behind an episode window.

**B. The "why pay?" constraint (a recurring user requirement).**
- Redesigned into a **fully local, free pipeline**: `mlx-whisper` ASR (Apple Silicon),
  BGE-M3 local embeddings, embedded Qdrant (no Docker), and free-tier Gemini/Groq for the
  answer step. Paid models are only needed when *better quality* is wanted, never to run.

**C. Deployment (Hugging Face Spaces).**
- `ModuleNotFoundError: 'app' is not a package` — the entry file `app.py` shadowed the
  `app` package. → Rename entry to `main.py`, delete the stale file.
- *429 (limit: 0) persisted after the fix* — HF **cached the pip layer** (unchanged
  `requirements.txt` → stale backend). → Add a cache-bust tag and pin `backend @main`.
- *Gradio version churn* — Gradio 6 dropped `type="messages"`; Gradio 4.44 clashed with
  newer `huggingface_hub` (HfFolder import). → Upgrade to 6.19 and drop the `type` arg.

**D. Provider quotas & blocks (the most time-consuming).**
- *All three free tiers exhausted/blocked* — Gemini 2.5-flash is 20/day and this account's
  2.0-flash quota was 0; the old Groq key returned 403 "Access denied" (key-bound regional
  block, **not** a network issue — a new key worked). → A `FallbackLLM` auto-demotes from a
  failing primary to a backup provider.
- *Graph build 413 "Request too large"* — a full episode (~21k tokens) exceeds Groq's 12k
  TPM. → Window each episode (6000 chars) + global throttle.
- *Graph build 429 lost all work* — the daily token budget ran out at episode 13/14 and the
  build only saved at the end. → **Incremental save per episode** + halve extractor
  `max_tokens`.
- *GeminiLLM crashed at construction* — `genai.Client` requires the key at build time. →
  Make the client **lazy** (deferred `_aio` property).

**E. Quality issues the user caught (product intuition, not tests).**
- *Citations pointing at 0:00 were actually the sponsor ad* — chunk 0 bundled the ad with
  real content. → A `_strip_leading_ad()` step drops the leading ~90s "本期節目由…" block (a
  reindex makes it take effect in the vector store).
- *Disclaimer wording changed every time* — it was LLM-generated. → Remove it from every
  strategy's system prompt; append a **fixed** disclaimer string in code.
- *Suggestions "secretly asked their own question"* — the feature auto-asked instead of
  offering choices. → Rebuild as **clickable buttons**: a vague query renders up to 3
  candidate questions; the user sees and picks one — fully transparent.
- *Suggestion content was off-topic* — Llama-70b read 記憶體 ("memory") as human memory and
  produced textbook questions. → Bake the **investing-domain context** into the prompt
  (記憶體 = memory/DRAM stocks), yielding grounded host-opinion questions.

**F. Self-inflicted.**
- `prepare_space.sh` deleted its own working directory when run from inside `space_build`.
  → Run it from the repo root, then `cd`.
- `git push` DNS failure (`Could not resolve host`) from a VPN/network hiccup. → Retried
  after the network recovered; everything pushed.

**One-line takeaway:** the hard part wasn't writing *a* RAG — it was (1) a contract strong
enough to make four strategies genuinely hot-swappable, (2) squeezing a working free demo
out of three simultaneously rate-limited providers, and (3) proving Corrective is best with
an eval harness rather than intuition. The most important quality fixes came from human
review, not the test suite — which is exactly why human review is irreplaceable in RAG.

## How this project maps to agentic-AI role requirements

An honest self-assessment against a set of AI-agent engineering requirements. ✅ = fully
demonstrated, 🟡 = partially demonstrated (with the gap stated), ⬜ = not built here.

- [x] ✅ **LLMs, modern AI frameworks, and/or ML.** Three LLM providers behind a
  fallback adapter, BGE-M3 embeddings, embedded Qdrant, local Whisper ASR, and a
  RAGAS-style eval harness. *(Gap: no RL environments — this is a retrieval/agent system,
  not RL.)*
- [x] ✅ **Prompt-engineering strategies.** Distinct, purpose-built prompts per strategy:
  agentic tool-use system prompt, Corrective RAG **grader** + **query-rewrite** prompts,
  domain-grounded **suggestion** prompt, and a fixed (non-generated) disclaimer. See
  [`corrective/strategy.py`](backend/app/rag/corrective/strategy.py) and
  [`suggest.py`](backend/app/rag/suggest.py).
- [x] 🟡 **Strong foundation in Python and TypeScript.** Python is the whole backend
  (RAG, ingestion, eval). TypeScript is a **minimal** Next.js chat frontend
  ([`frontend/app/page.tsx`](frontend/app/page.tsx), `tsconfig.json`) — present but small.
- [x] ✅ **Designing tool-calling environments to evaluate & benchmark agent systems.**
  A 4-tool retrieval environment (`vector_search`, `keyword_search`, `expand_context`,
  `get_episode_metadata`) the agent calls in a loop, plus a benchmark harness that scores
  and **compares three agent strategies** on a golden set (this is how "Corrective is best"
  was established). See [`agentic/toolbox.py`](backend/app/rag/agentic/toolbox.py) and
  [`eval/`](backend/app/eval/).
- [x] 🟡 **Agentic systems that predict and execute users' next steps.** A query
  suggester **predicts** the user's likely next questions (grounded in retrieved content)
  and renders them as clickable actions that **execute** on click
  ([`suggest.py`](backend/app/rag/suggest.py)). *(Gap: it predicts the next question, not
  multi-step actions across a complex workflow.)*
- [ ] ⬜ **Mapping user paths on real-world software to API functionality and action
  trajectories.** Not built — no UI-path-to-API mapping here.
- [x] 🟡 **A searchable, self-updating memory store for continuously learning agents.**
  The corpus is a **searchable** vector store that **self-updates** from the RSS feed and
  trims to recent episodes ([`scripts/update_corpus.py`](backend/scripts/update_corpus.py),
  `corpus_range()`). *(Gap: it's a refreshing knowledge store, not per-agent episodic
  memory that learns from each interaction.)*
- [ ] ⬜ **Interpreting DOM snapshots, mouse clicks, and keyboard inputs to select browser
  actions.** Not built — there is no browser-automation / computer-use layer.
- [x] 🟡 **A context composer that feeds relevant info into prompts from interactions,
  content, and memory.** Each RAG strategy *is* a context composer: it retrieves the
  relevant passages from the store and composes them into the LLM prompt, and adapts when a
  query is ambiguous. *(Gap: composes from retrieval + the corpus, not from live page
  content or browser-interaction signals.)*

**Summary:** strong on LLM/prompt-engineering, tool-calling agent environments, and
benchmarking (✅); partial on next-step prediction, a self-updating searchable store, and
context composition (🟡); the browser/DOM-action and UI-path→API-mapping pieces are out of
scope here (⬜).

## Status
M0–M3 complete + Corrective RAG; 3 strategies ready. **EP658–EP671 (14 episodes)** built
and searchable for free, auto-updating to the most recent episodes. Tier-1 retrieval
scorecard done (hybrid wins). Next: full answer-quality comparison (M4) once LLM budget
allows.
</content>
</invoke>
