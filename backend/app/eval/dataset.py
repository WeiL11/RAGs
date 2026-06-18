"""Golden Q&A dataset loader for the RAG evaluation harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import REPO_ROOT

GOLDEN_PATH = REPO_ROOT / "eval" / "golden.jsonl"


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: str
    gold_episodes: list[str] = field(default_factory=list)
    reference: str = ""


def load_golden(path: str | Path | None = None) -> list[EvalQuestion]:
    p = Path(path or GOLDEN_PATH)
    out: list[EvalQuestion] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(EvalQuestion(**json.loads(line)))
    return out
