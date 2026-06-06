# results/

Evaluation outputs, organized so repeated runs never overwrite each other.

```
results/
├── runs/         one subdirectory per run: <YYYY-MM-DD_HHMMSS_model>/
├── comparisons/  timestamped aggregate snapshots (process history, never overwritten)
├── tables/       hand-picked final tables for the report/paper
└── figures/      plots (GA/GN/QR bars, WEFAT heatmaps, ...)
```

## runs/

Every evaluation creates a fresh, timestamped subdirectory — e.g.
`2026-06-03_211530_mistral/` — so running many times a day is safe and
traceable. Each run directory contains:

| file               | what it is                                            |
|--------------------|-------------------------------------------------------|
| `eval_detail.csv`  | one row per query: query, output, GA, GN, QR, 2nd-pass |
| `eval_summary.csv` | this run's aggregate GA/GN/QR (%)                      |
| `wefat.csv`        | WEFAT scores (only if the WEFAT script was run)        |
| `queries.txt`      | snapshot of the exact input used                       |
| `meta.json`        | run config: model, temperature, top_k, n, full timestamp |

Directory name format `YYYY-MM-DD_HHMMSS_<model>`: dates sort chronologically by
filename, and the model name is visible without opening anything.

All numbers in a run directory are produced on this machine in that run. Paper
figures are NOT stored here — if you want to compare against the paper, keep
those numbers in your own notes/report, not mixed into measured results.

## comparisons/

`aggregate_runs.py` collects the per-run summaries into one table. By default it
writes a **timestamped** file here — `comparison_<YYYY-MM-DD_HHMMSS>_<tag>.csv` —
so every aggregation is kept and nothing is overwritten. This is your process
history: you can see how the comparison looked at each point.

```bash
python scripts/aggregate_runs.py --tag det     # -> comparisons/comparison_<ts>_det.csv
```

## tables/

For the curated FINAL table that goes into the report. `aggregate_runs.py` only
writes here when you pass an explicit `--out`:

```bash
python scripts/aggregate_runs.py --tag det --out final_model_comparison.csv
```

Keep this directory small — just the versions you actually cite.

## figures/

Reserved for plots. Empty for now.

## Note

Results depend on the base model, the judge, temperature (0.7 = stochastic), and
the prompt set. Small sets and a single run vary between runs — see `../NOTES.md`.

## Version control

`runs/` is typically git-ignored (large, regenerable). `tables/` and `figures/`
are kept, since those are the curated deliverables.
