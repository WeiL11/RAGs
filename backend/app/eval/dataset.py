"""Golden Q&A dataset loader for the RAG evaluation harness.

Prefers the human-editable YAML (`eval/golden.yaml`); falls back to `eval/golden.jsonl`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from app.config import REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval"
GOLDEN_YAML = EVAL_DIR / "golden.yaml"
GOLDEN_JSONL = EVAL_DIR / "golden.jsonl"


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: str
    gold_episodes: list[str] = field(default_factory=list)
    reference: str = ""


def _coerce(d: dict) -> EvalQuestion:
    known = {f.name for f in fields(EvalQuestion)}
    return EvalQuestion(**{k: v for k, v in d.items() if k in known})


def load_golden(path: str | Path | None = None) -> list[EvalQuestion]:
    p = Path(path) if path else (GOLDEN_YAML if GOLDEN_YAML.exists() else GOLDEN_JSONL)
    if p.suffix in (".yaml", ".yml"):
        import yaml  # lazy

        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or []
        items = raw.get("questions", []) if isinstance(raw, dict) else raw
        return [_coerce(d) for d in items]
    # JSONL
    out: list[EvalQuestion] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(_coerce(json.loads(line)))
    return out
