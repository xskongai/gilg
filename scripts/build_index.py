"""Build the FAISS index from the raw spreadsheets (run once per language).

Usage:
    python scripts/build_index.py            # English (default)
    python scripts/build_index.py --lang zh  # Chinese
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))            # project root, for `import config`
sys.path.insert(0, str(_ROOT / "src"))    # for `import gilg`

from gilg.retrieval.vector_store import build_index  # noqa: E402
from config.config import cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, choices=["en", "zh"],
                        help="language profile (switches data + embedding + index dir)")
    args = parser.parse_args()

    if args.lang:
        from config.config import set_lang
        set_lang(args.lang)

    store = build_index(save=True)
    print(f"[{cfg.prompt.lang}] Built FAISS index with {store.index.ntotal} vectors "
          f"at {cfg.retrieval.index_path}")


if __name__ == "__main__":
    main()
