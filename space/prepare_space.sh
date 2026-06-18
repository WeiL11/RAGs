#!/usr/bin/env bash
# Assemble a self-contained folder to push to a Hugging Face Space:
#   app.py + requirements.txt + README.md + the prebuilt index under data/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/space_build}"

rm -rf "$OUT"
mkdir -p "$OUT/data"
cp "$ROOT/space/main.py" "$ROOT/space/requirements.txt" "$ROOT/space/README.md" "$OUT/"
cp -R "$ROOT/data/transcripts" "$OUT/data/transcripts"
cp -R "$ROOT/data/qdrant_local" "$OUT/data/qdrant_local"
if [ -f "$ROOT/data/graph.json" ]; then cp "$ROOT/data/graph.json" "$OUT/data/graph.json"; fi

echo "Built Space folder: $OUT"
du -sh "$OUT"
echo "Contents:"; ls -1 "$OUT"
