---
title: Gooaye RAG
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.19.0
app_file: main.py
pinned: false
---

# Gooaye (股癌) Podcast RAG — demo

Ask questions about the Gooaye 股癌 podcast; answers are grounded in episode transcripts
and cite the source episode + timestamp. Free stack: local BGE-M3 embeddings + a prebuilt
Qdrant index (shipped under `data/`) + Google Gemini for the answer.

**Setup:** add `GEMINI_API_KEY` under Settings → Secrets. First boot downloads the BGE-M3
model (~2 GB), so the initial start takes a few minutes.

Not investment advice — the content reflects the show's personal opinions.
