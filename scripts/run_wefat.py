"""Run WEFAT on pipeline outputs. Writes results/runs/<timestamp_model>/wefat.csv.

Lower |score| = less gender bias. Requires data/glove.6B.100d.txt.
NOTE: WEFAT here is known to be methodologically flawed — see
src/gilg/evaluation/wefat.py and NOTES.md. Treat numbers with caution.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from gilg.evaluation.wefat import load_glove, response_wefat  # noqa: E402
from gilg.generation.pipeline import Pipeline  # noqa: E402
from gilg.utils.run_dir import new_run_dir, write_meta  # noqa: E402
from gilg.utils.seeding import set_seed  # noqa: E402
from config.config import cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="file with one query per line")
    parser.add_argument("--tag", default="wefat", help="label appended to the run dir")
    args = parser.parse_args()

    set_seed(cfg.seed)
    queries = [q.strip() for q in Path(args.file).read_text().splitlines() if q.strip()]

    print("Loading GloVe embeddings (~10s)...")
    emb = load_glove()

    run_dir = new_run_dir(tag=args.tag)
    pipeline = Pipeline()

    rows = []
    scores = []
    for q in queries:
        r = pipeline.run(q)
        score = response_wefat(r.r_final, emb)
        scores.append(score)
        rows.append((q, r.r_final, f"{score:.4f}"))
        print(f"{score:.4f}  <- {r.r_final[:70]}")

    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"\nMean |WEFAT| over {len(scores)} responses: {mean:.4f}")
    print("(lower = less gender bias; near 0 = neutral)")

    with open(run_dir / "wefat.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "output", "abs_wefat"])
        w.writerows(rows)

    shutil.copyfile(args.file, run_dir / "queries.txt")
    write_meta(run_dir, metric="wefat", n_queries=len(scores), mean_abs_wefat=mean)

    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
