# Experiment Design — Which RAG performs best?

Goal: rank the three strategies — **Agentic**, **Graph**, **Corrective** — on the
Gooaye corpus, objectively and reproducibly. This doc is the protocol; the runner
lives in `backend/scripts/eval_retrieval.py` (free tier) and `app/eval/` (full tier).

## 1. Principle: separate "did it find the right thing" from "did it write a good answer"

RAG quality has two independent failure points:
- **Retrieval** — were the *right passages* fetched? (the "R")
- **Generation** — given good passages, was the *answer* faithful and useful? (the "G")

We measure them separately so a bad answer can be blamed on the right stage. This also
gives a **free tier** (retrieval needs no LLM) and a **paid tier** (generation needs one).

## 2. The dataset (golden Q&A set)

File: `eval/golden.jsonl` — one question per line (the actual questions are in Chinese,
since that's the podcast's language; shown here in English only to illustrate the schema):
```json
{"id":"q1","question":"Where does the name 'Gooaye' come from?","category":"lookup",
 "gold_episodes":["EP665"],"reference":"From a TV character named Noah…"}
```
Fields: `category`, `gold_episodes` (which episodes truly contain the answer — used for
free retrieval scoring), `reference` (a human reference answer — used for paid scoring).

**Size:** start ~20–30 questions, grow to ~50. **Balance by category** so each strategy's
strengths/weaknesses show up:

| Category | Example (asked in Chinese) | Expected to favor |
|---|---|---|
| `lookup` | "What sponsors does the show have?" | all (esp. keyword) |
| `opinion` | "What's his recent view on US stocks?" | Agentic |
| `aggregation` | "Summarize his views on AI servers" | Graph |
| `relationship` | "How does he connect NVIDIA and TSMC's supply chain?" | Graph |
| `multi_hop` | "That company he mentioned — what's his later take?" | Agentic / Corrective |
| `negative` | "His view on Bitcoin?" (not in corpus) | tests honest "I don't know" |

The `negative` rows are crucial: they catch hallucination — a good system admits it has
no answer, a bad one invents one.

## 3. Metrics

### Tier 1 — Retrieval quality (FREE, no LLM)
For each question, run the retriever and compare retrieved episodes to `gold_episodes`:
- **Recall@k** — fraction of gold episodes that appear in the top-k. (Most important.)
- **Hit@k** — did *any* gold episode appear? (1/0)
- **MRR** — 1/rank of the first gold hit (rewards ranking the right thing high).
- **Precision@k** — fraction of retrieved that are gold (noise check).
Compared across retrievers: **vector**, **keyword (BM25)**, **hybrid**, and **graph**
(graph needs the graph built — paid one-time).

### Tier 2 — Generation quality (PAID, needs Claude)
Run each full strategy end-to-end, then score with **RAGAS** + reference:
- **Faithfulness** — is every claim supported by retrieved context? (anti-hallucination)
- **Answer relevancy** — does the answer address the question?
- **Context recall / precision** — RAGAS's view of retrieval vs the reference.
- **Answer correctness** — vs the human `reference`.

### Tier 3 — Operational (captured in every `trace`, free to collect)
- **latency_ms**, **cost_usd**, **tokens**, **#LLM calls / tool_calls / rounds**.
A strategy that wins on quality but costs 5× and is 3× slower may still lose for production.

## 4. Protocol (fair comparison)
1. **Same everything**: same corpus, same embedder, same `k`, same questions, same judge
   model (use a fixed model, e.g. `claude-opus-4-8`, as the RAGAS judge for all).
2. Run every question through every strategy; log all `trace`s + retrieved contexts +
   answers to `data/eval/results/<timestamp>.jsonl`.
3. Aggregate into a **scorecard**: mean per metric, **overall and per-category**.
4. Report **variance** (run the set ≥2×, or bootstrap) — Tier-2 LLM scores are noisy;
   a 2-point gap inside the noise band is not a win.
5. **Decision rule:** pick by the metric that matches your use — faithfulness for trust,
   answer-correctness for accuracy, then break ties on cost/latency. Expect *no single
   winner*: likely Agentic for opinion/multi-hop, Graph for aggregation/relationship,
   Corrective best on noisy/negative cases. The output is a **per-category recommendation**,
   not one champion.

## 5. Honesty guards
- **Don't label gold by running the system** (circular). Label `gold_episodes` from the
  transcripts/your knowledge, independently.
- **Blind the judge** to which strategy produced an answer.
- **Log every cap** (top-k, max rounds) so "missed" isn't confused with "not retrieved".
- Keep the dataset in git; version it as it grows so scores stay comparable over time.

## 6. How to run
```bash
# FREE — retrieval quality of vector vs keyword vs hybrid (no key):
python scripts/eval_retrieval.py

# PAID — full end-to-end RAGAS comparison of all strategies (needs ANTHROPIC_API_KEY):
python -m app.eval.run_eval        # (M4 — builds on the same dataset)
```

## 7. Status
- ✅ Three strategies implemented (agentic, graph, corrective).
- ✅ Free retrieval-eval runner + **21-question golden set** (6 categories), with
  gold episodes grounded in the corpus by term frequency. `reference` answers are
  brief stubs — fill them in before running Tier-2.
- ✅ Tier-1 result (19 scored, k=8): **hybrid** best Recall@8 **0.87** (vector 0.75,
  keyword 0.68) and best per category; vector best MRR among single retrievers.
- ⏳ Tier-2 RAGAS runner (`app/eval/run_eval.py`) — needs a key; that's M4.
- 🔜 Grow to ~30 questions and have a human verify gold + references.
