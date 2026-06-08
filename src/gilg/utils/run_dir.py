"""Create timestamped run directories under results/runs/.

Each evaluation run gets its own folder named <YYYY-MM-DD_HHMMSS_model>, so
repeated runs (even many per day) never overwrite each other and stay traceable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config.config import ROOT, active_model_name, cfg


def new_run_dir(tag: str | None = None) -> Path:
    """Make and return a fresh run directory.

    Name: results/runs/YYYY-MM-DD_HHMMSS_<model>[_<tag>]/
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    model = active_model_name().replace("/", "-").replace(":", "-")
    name = f"{stamp}_{model}" + (f"_{tag}" if tag else "")
    path = ROOT / "results" / "runs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_meta(run_dir: Path, **extra) -> None:
    """Write meta.json capturing the run configuration."""
    meta = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "lang": cfg.prompt.lang,
        "backend": cfg.generation.backend,
        "model": active_model_name(),
        "temperature": cfg.generation.temperature,
        "top_p": cfg.generation.top_p,
        "max_new_tokens": cfg.generation.max_new_tokens,
        "top_k": cfg.retrieval.top_k,
        "embedding_model": cfg.retrieval.embedding_model,
        "judge_model": cfg.evaluation.judge_model,
        "seed": cfg.seed,
        **extra,
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
