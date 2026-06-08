"""Evaluate pipeline outputs with GA/GN/QR (LLM-as-Judge).

Each run writes to a fresh timestamped dir results/runs/<YYYY-MM-DD_HHMMSS_model>/:
  - eval_detail.csv   one row per query (query, output, GA, GN, QR, 2nd-pass)
  - eval_summary.csv  this run's aggregate GA/GN/QR (%)
  - queries.txt       snapshot of the input
  - meta.json         run config

All numbers here are produced on THIS machine in THIS run. No external/paper
figures are mixed in.

Usage:
    python scripts/run_evaluation.py --file queries.txt
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))            # project root, for `import config`
sys.path.insert(0, str(_ROOT / "src"))    # for `import gilg`

from gilg.evaluation.judge import Judge  # noqa: E402
from gilg.evaluation.metrics import aggregate  # noqa: E402
from gilg.generation.pipeline import Pipeline  # noqa: E402
from gilg.utils.run_dir import new_run_dir, write_meta  # noqa: E402
from gilg.utils.seeding import set_seed  # noqa: E402
from config.config import cfg, active_model_name  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="file with one query per line")
    parser.add_argument("--lang", default=None, choices=["en", "zh"],
                        help="language profile: switches data, embedding, model, index, prompts")
    parser.add_argument("--model", default=None, help="model name; cloud (gpt-4o, gemini-2.5-flash, deepseek-chat, qwen-max) or local Ollama (mistral, qwen2.5, qwen2.5:0.5b, llama3.2:1b). Overrides config/lang default")
    parser.add_argument("--temperature", type=float, default=None,
                        help="generation temperature (0 = deterministic/reproducible; default 0.7)")
    parser.add_argument("--tag", default=None, help="optional label appended to the run dir")
    parser.add_argument("--baseline", action="store_true",
                        help="zero-shot baseline (same model, no RAG/verifier/CoT) instead of the full pipeline")
    args = parser.parse_args()

    if args.lang:
        from config.config import set_lang
        set_lang(args.lang)
    if args.model:
        from config.config import set_model
        set_model(args.model)
    if args.temperature is not None:
        from config.config import set_temperature
        set_temperature(args.temperature)

    set_seed(cfg.seed)
    queries = [q.strip() for q in Path(args.file).read_text().splitlines() if q.strip()]

    run_dir = new_run_dir(tag=args.tag or ("baseline" if args.baseline else None))

    if args.baseline:
        from gilg.generation.baseline import Baseline
        pipeline = Baseline()
    else:
        pipeline = Pipeline()
    judge = Judge()

    rows = []
    scores = []
    for q in queries:
        r = pipeline.run(q)
        s = judge.score(q, r.r_final)
        scores.append(s)
        rows.append((q, r.r_final, s.ga, s.gn, s.qr, r.used_second_pass))
        print(f"{q}\n  -> {r.r_final}\n  GA={s.ga} GN={s.gn} QR={s.qr}")

    summary = aggregate(scores)
    print("\n=== Aggregate (%) ===")
    print(f"GA: {summary['ga']:.2f}  GN: {summary['gn']:.2f}  QR: {summary['qr']:.2f}")

    # per-query detail
    with open(run_dir / "eval_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "output", "GA", "GN", "QR", "used_second_pass"])
        w.writerows(rows)

    # this run's aggregate only — nothing external mixed in
    with open(run_dir / "eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "GA (%)", "GN (%)", "QR (%)"])
        w.writerow([
            active_model_name(), len(scores),
            f"{summary['ga']:.2f}", f"{summary['gn']:.2f}", f"{summary['qr']:.2f}",
        ])

    # input snapshot + run config
    shutil.copyfile(args.file, run_dir / "queries.txt")
    write_meta(run_dir, n_queries=len(scores), aggregate=summary, baseline=args.baseline)

    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()


