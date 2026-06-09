"""Aggregate evaluation runs into comparison tables.

Scans results/runs/, reads each run's eval_summary.csv + meta.json (+ the
queries.txt snapshot to detect language), and emits:

  1. A wide terminal table (one row per model, baseline vs proposed side by
     side) for quick at-a-glance analysis.
  2. A LaTeX booktabs table (Method stacked vertically as Baseline / Proposed,
     best value per metric bolded, Proposed row showing the gain inline,
     blocks separated by language) ready to paste into the paper.

Every number comes from a real run on this machine. Nothing external is added.

Multiple runs of the same (lang, model, mode): only the LATEST is used
(timestamp wins). Use --tag to scope to one batch of runs.

Usage:
    python scripts/aggregate_runs.py                 # all runs
    python scripts/aggregate_runs.py --tag cmp       # only runs tagged _cmp
    python scripts/aggregate_runs.py --out final     # curated copy to tables/
    python scripts/aggregate_runs.py --no-gain       # hide inline (+delta)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

RUNS_DIR = _ROOT / "results" / "runs"
TABLES_DIR = _ROOT / "results" / "tables"
COMPARISONS_DIR = _ROOT / "results" / "comparisons"

PRETTY = {
    "qwen2.5": "Qwen2.5-7B",
    "glm4:9b": "GLM-4-9B",
    "yi:9b": "Yi-1.5-9B",
    "mistral": "Mistral-7B",
    "llama3.1": "Llama-3.1-8B",
    "gemma2:9b": "Gemma-2-9B",
    "gpt-4o": "GPT-4o",
    "gemini-3.5-flash": "Gemini-3.5-Flash",
    "deepseek-chat": "DeepSeek-V3",
    "qwen3.7-max": "Qwen3.7-Max",
}
LANG_NAME = {"en": "English", "zh": "Chinese", "?": "Unknown"}
METRICS = ("ga", "gn", "qr")

# Fixed display order: big (cloud) models first, then small (local) models.
# Models not listed here sort last, alphabetically. Used for both tables.
MODEL_ORDER = [
    # big / cloud
    "gpt-4o", "gemini-3.5-flash", "deepseek-chat", "qwen3.7-max",
    # small / local — English
    "mistral", "llama3.1", "gemma2:9b",
    # small / local — Chinese
    "qwen2.5", "glm4:9b", "yi:9b",
]
_ORDER_INDEX = {m: i for i, m in enumerate(MODEL_ORDER)}


def _model_key(model: str):
    """Sort key: listed models by MODEL_ORDER, others after, alphabetical."""
    return (_ORDER_INDEX.get(model, len(MODEL_ORDER)), model)


def _detect_lang(run_dir: Path) -> str:
    q = run_dir / "queries.txt"
    if q.exists():
        text = q.read_text(encoding="utf-8", errors="ignore")
        return "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "en"
    return "?"


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "eval_summary.csv"
    if not summary_path.exists():
        return None
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    if not rows:
        return None
    row = rows[0]

    model = row.get("model", "?")
    mode = "proposed"
    timestamp = ""
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.load(meta_path.open(encoding="utf-8"))
            model = meta.get("model", model)
            timestamp = meta.get("timestamp", "")
            if meta.get("baseline"):
                mode = "baseline"
        except Exception:
            pass
    if not timestamp:
        timestamp = run_dir.name[:17]
    if not meta_path.exists() and run_dir.name.endswith("_baseline"):
        mode = "baseline"

    return {
        "lang": _detect_lang(run_dir),
        "model": model,
        "mode": mode,
        "n": row.get("n", ""),
        "ga": _to_float(row.get("GA (%)")),
        "gn": _to_float(row.get("GN (%)")),
        "qr": _to_float(row.get("QR (%)")),
        "timestamp": timestamp,
        "run_dir": run_dir.name,
    }


def _f(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "-"


def _latex_num(v, *, bold=False, gain=None):
    if not isinstance(v, (int, float)):
        return "-"
    s = f"{v:.1f}"
    if gain is not None:
        s += f"\\,({gain:+.1f})"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def build_latex(pivot, langs, show_gain):
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\begin{tabular}{llccc}", "\\toprule",
             "Model & Method & GA & GN & QR \\\\"]
    for lang in langs:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{LANG_NAME.get(lang, lang)}}}}} \\\\")
        lines.append("\\midrule")
        models = sorted((m for (lg, m) in pivot if lg == lang), key=_model_key)
        for mi, model in enumerate(models):
            p = pivot[(lang, model)]
            b, pr = p.get("baseline"), p.get("proposed")
            pretty = PRETTY.get(model, model)
            cells_b, cells_p = [], []
            for m in METRICS:
                bv = b[m] if b else None
                pv = pr[m] if pr else None
                bbold = isinstance(bv, (int, float)) and isinstance(pv, (int, float)) and bv >= pv
                pbold = isinstance(bv, (int, float)) and isinstance(pv, (int, float)) and pv >= bv
                gain = (pv - bv) if (show_gain and isinstance(bv, (int, float)) and isinstance(pv, (int, float))) else None
                cells_b.append(_latex_num(bv, bold=bbold))
                cells_p.append(_latex_num(pv, bold=pbold, gain=gain))
            lines.append(f"\\multirow{{2}}{{*}}{{{pretty}}} "
                         f"& Baseline & {cells_b[0]} & {cells_b[1]} & {cells_b[2]} \\\\")
            lines.append(f"& Proposed & {cells_p[0]} & {cells_p[1]} & {cells_p[2]} \\\\")
            if mi != len(models) - 1:
                lines.append("\\cmidrule(lr){1-5}")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    cap = ("Gender-inclusive generation results (\\%). GA: gender-assumption, "
           "GN: gender-neutrality, QR: quality-relevance. Judge: GPT-4o, "
           "temperature 0. Best per metric in \\textbf{bold}"
           + ("; Proposed shows gain over Baseline in parentheses." if show_gain else "."))
    lines += [f"\\caption{{{cap}}}", "\\label{tab:results}", "\\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gain", action="store_true")
    args = ap.parse_args()

    if not RUNS_DIR.exists():
        sys.exit(f"No runs directory at {RUNS_DIR}")

    # Only timestamped run dirs (YYYY-MM-DD_...); skip history/, tables/, etc.
    _SKIP = {"history", "tables", "comparisons", "archive", "old"}
    run_dirs = sorted(
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and d.name not in _SKIP and d.name[:4].isdigit()
    )
    if args.tag:
        run_dirs = [d for d in run_dirs if d.name.endswith(f"_{args.tag}")]

    records = [r for d in run_dirs if (r := read_run(d))]
    if not records:
        sys.exit("No matching runs with an eval_summary.csv found.")

    latest = {}
    for r in records:
        key = (r["lang"], r["model"], r["mode"])
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r

    pivot = {}
    for r in latest.values():
        pivot.setdefault((r["lang"], r["model"]), {})[r["mode"]] = r

    langs = sorted({lg for (lg, _m) in pivot})
    show_gain = not args.no_gain

    print("\n" + "=" * 78)
    print("WIDE TABLE - baseline vs proposed (for analysis)"
          + (f"   [tag={args.tag}]" if args.tag else ""))
    print("=" * 78)
    h = (f'{"lang":<5}{"model":<16}'
         f'{"GA_b":>8}{"GA_p":>8}{"GN_b":>8}{"GN_p":>8}{"QR_b":>8}{"QR_p":>8}')
    print(h)
    print("-" * len(h))
    for (lang, model) in sorted(pivot, key=lambda k: (k[0], _model_key(k[1]))):
        p = pivot[(lang, model)]
        b, pr = p.get("baseline"), p.get("proposed")
        print(f'{lang:<5}{model:<16}'
              f'{_f(b["ga"] if b else None):>8}{_f(pr["ga"] if pr else None):>8}'
              f'{_f(b["gn"] if b else None):>8}{_f(pr["gn"] if pr else None):>8}'
              f'{_f(b["qr"] if b else None):>8}{_f(pr["qr"] if pr else None):>8}')

    latex = build_latex(pivot, langs, show_gain)
    print("\n" + "=" * 78)
    print("LATEX TABLE - paste into paper")
    print("=" * 78)
    print(latex)

    # Output paths. Both --out (curated, tables/) and default (comparisons/)
    # get a timestamp so nothing is ever overwritten.
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if args.out:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        stem = args.out[:-4] if args.out.endswith(".csv") else args.out
        base = TABLES_DIR / f"{stem}_{stamp}"
    else:
        COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
        sfx = f"_{args.tag}" if args.tag else ""
        base = COMPARISONS_DIR / f"{stamp}{sfx}"

    wide_path = Path(f"{base}_wide.csv")
    long_path = Path(f"{base}_long.csv")
    tex_path = Path(f"{base}.tex")

    ordered = sorted(pivot, key=lambda k: (k[0], _model_key(k[1])))

    # Wide CSV: one row per model, baseline vs proposed side by side.
    with wide_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang", "model", "GA_baseline", "GA_proposed",
                    "GN_baseline", "GN_proposed", "QR_baseline", "QR_proposed"])
        for (lang, model) in ordered:
            p = pivot[(lang, model)]
            b, pr = p.get("baseline"), p.get("proposed")
            w.writerow([lang, model,
                        _f(b["ga"] if b else None), _f(pr["ga"] if pr else None),
                        _f(b["gn"] if b else None), _f(pr["gn"] if pr else None),
                        _f(b["qr"] if b else None), _f(pr["qr"] if pr else None)])

    # Long CSV: one row per (model, method) — matches the paper table layout.
    with long_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lang", "model", "method", "GA", "GN", "QR"])
        for (lang, model) in ordered:
            p = pivot[(lang, model)]
            for method, key in (("baseline", "baseline"), ("proposed", "proposed")):
                r = p.get(key)
                if r:
                    w.writerow([lang, model, method, _f(r["ga"]), _f(r["gn"]), _f(r["qr"])])

    tex_path.write_text(latex + "\n", encoding="utf-8")
    print(f"\nSaved:\n  {wide_path}\n  {long_path}\n  {tex_path}")


if __name__ == "__main__":
    main()