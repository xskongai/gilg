"""Run the two-pass inclusive-generation pipeline.

Usage:
    python scripts/run_pipeline.py "Most successful CEOs are ___"
    python scripts/run_pipeline.py --file queries.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))            # project root, for `import config`
sys.path.insert(0, str(_ROOT / "src"))    # for `import gilg`

from gilg.generation.pipeline import Pipeline  # noqa: E402
from gilg.utils.seeding import set_seed  # noqa: E402
from config.config import cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="a single query")
    parser.add_argument("--file", help="file with one query per line")
    parser.add_argument("--lang", default=None, choices=["en", "zh"],
                        help="language profile: switches data, embedding, model, index, prompts")
    parser.add_argument("--model", default=None, help="Ollama model to use (overrides config/lang default)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="generation temperature (0 = deterministic; default 0.7)")
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

    if args.file:
        queries = [q.strip() for q in Path(args.file).read_text().splitlines() if q.strip()]
    elif args.query:
        queries = [args.query]
    else:
        parser.error("provide a query or --file")

    pipeline = Pipeline()
    for q in queries:
        r = pipeline.run(q)
        print(f"\nQuery: {r.query}")
        print(f"R1:    {r.r1}")
        print(f"Final: {r.r_final}")
        print(f"(second pass used: {r.used_second_pass})")


if __name__ == "__main__":
    main()
