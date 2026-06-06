"""Aggregate multiple evaluation runs into one comparison table.

Scans results/runs/ for run directories (optionally filtered by --tag), reads
each one's eval_summary.csv, and writes a combined table to
results/tables/model_comparison.csv.

Every number comes from a real run on this machine. Nothing external is added.

Usage:
    python scripts/aggregate_runs.py --tag cmp
    python scripts/aggregate_runs.py                 # all runs
    python scripts/aggregate_runs.py --out my_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

RUNS_DIR = _ROOT / "results" / "runs"
TABLES_DIR = _ROOT / "results" / "tables"
COMPARISONS_DIR = _ROOT / "results" / "comparisons"


def _read_summary(run_dir: Path) -> dict | None:
    """Read one run's summary row + model/timestamp from meta.json."""
    summary_path = run_dir / "eval_summary.csv"
    meta_path = run_dir / "meta.json"
    if not summary_path.exists():
        return None

    with open(summary_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]

    model, timestamp = row.get("model", "?"), ""
    if meta_path.exists():
        meta = json.load(open(meta_path, encoding="utf-8"))
        model = meta.get("model", model)
        timestamp = meta.get("timestamp", "")

    return {
        "model": model,
        "n": row.get("n", ""),
        "GA (%)": row.get("GA (%)", ""),
        "GN (%)": row.get("GN (%)", ""),
        "QR (%)": row.get("QR (%)", ""),
        "run_dir": run_dir.name,
        "timestamp": timestamp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="only include run dirs whose name ends with _<tag>")
    parser.add_argument("--out", default=None,
                        help="explicit output path under results/tables/ (final, curated). "
                             "If omitted, a timestamped file is written to results/comparisons/ (never overwrites).")
    args = parser.parse_args()

    if not RUNS_DIR.exists():
        sys.exit(f"No runs directory at {RUNS_DIR}")

    run_dirs = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir())
    if args.tag:
        run_dirs = [d for d in run_dirs if d.name.endswith(f"_{args.tag}")]

    records = [r for d in run_dirs if (r := _read_summary(d))]
    if not records:
        sys.exit("No matching runs with an eval_summary.csv found.")

    # Default: timestamped file in comparisons/ so process snapshots accumulate
    # and nothing is overwritten. --out goes to tables/ for a curated final copy.
    if args.out:
        out_path = TABLES_DIR / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        suffix = f"_{args.tag}" if args.tag else ""
        COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = COMPARISONS_DIR / f"comparison_{stamp}{suffix}.csv"

    cols = ["model", "n", "GA (%)", "GN (%)", "QR (%)", "run_dir", "timestamp"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)

    # also print a readable table
    print(f"{'model':<12}{'n':>4}{'GA(%)':>9}{'GN(%)':>9}{'QR(%)':>9}")
    for r in records:
        print(f"{r['model']:<12}{r['n']:>4}{r['GA (%)']:>9}{r['GN (%)']:>9}{r['QR (%)']:>9}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
