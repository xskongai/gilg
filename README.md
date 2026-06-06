# GILG — Gender-Inclusive Language Generation (refactored)

A clean, modular reimplementation of the two-pass **RAG + CoT** inclusive-language
framework from *"Gender inclusive language generation framework: A reasoning
approach with RAG and CoT"* (Knowledge-Based Systems, 2025). The original code
shipped as standalone Colab notebooks; this is the same method reorganized into a
maintainable Python project with externalized prompts and a single config.

## Layout

```
config/        single source of truth (models, paths, hyperparameters)
prompts/       all prompts as Jinja2 .j2 files; en/ now, zh/ later (parallel)
data/raw/      the three source spreadsheets (693 pairs / 125 rules / 726 pairs)
data/index/    persisted FAISS index (built once)
src/gilg/
  data/        load spreadsheets -> structured Documents
  retrieval/   one embedding model, FAISS build/load, top-k retriever
  generation/  first_pass, verifier, second_pass, and pipeline (the two-pass flow)
  evaluation/  LLM-as-Judge (GA/GN/QR), WEFAT, metric aggregation
  utils/       Jinja2 prompt rendering, seeding, io
scripts/       build_index / run_pipeline / run_evaluation
tests/         loader, prompt-rendering, pipeline-wiring tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env        # OPENAI_API_KEY (only needed for evaluation)
```

### Local model via Ollama (default backend)

Generation runs a local Mistral-7B-Instruct through [Ollama](https://ollama.com)
— free, reproducible, and fast on Apple Silicon. HF Hub no longer serves 7B
models for free, so this replaces the notebooks' HF Hub calls.

```bash
# install Ollama (macOS): https://ollama.com/download  (or: brew install ollama)
ollama pull mistral         # one-time model download (~4 GB, 4-bit quantized)
```

Ollama serves automatically in the background. To use a different model, set
`cfg.generation.ollama_model` in `config/config.py` (e.g. "llama3.1:8b").

## Usage

```bash
python scripts/build_index.py                       # build FAISS once
python scripts/run_pipeline.py "A ___ is caring."    # single query
python scripts/run_evaluation.py --file queries.txt  # GA/GN/QR over a set
```

## Switching prompts / language

Prompts live in `prompts/en/`. To run a Chinese version later, add
`prompts/zh/` with the same filenames and set `cfg.prompt.lang = "zh"` — no code
changes. Individual templates are selected in `config/config.py`.

## Notes on faithfulness to the original

This refactor preserves the experiment but fixes things that were broken or
inconsistent in the source. Documented deliberately so results stay traceable:

- **One embedding model.** The notebooks mixed `all-MiniLM-L6-v2` (384-d) and
  `all-mpnet-base-v2` (768-d) across passes, giving incompatible indexes.
  We use a single model (`all-mpnet-base-v2` by default, set in config).
- **No xlsx -> PDF detour.** The original exported data to PDF and re-split it by
  character count, scrambling pair mappings. We build one structured Document per
  pair/rule instead, with metadata.
- **CoT prompt = the version that actually ran** (from `Second Pass.ipynb`), with
  the decorative emoji bullets replaced by plain numerals. All wording — including
  the original's quirks — is otherwise unchanged, kept in `prompts/en/cot_rewrite.j2`.
- **Verifier** reuses the Gender-Neutrality (GN) criterion (0 = non-inclusive,
  1 = inclusive); there was no standalone verifier prompt in the source.
- **WEFAT** ports the paper's association score (Eq. 1) faithfully but does **not**
  replicate the notebook's regression-against-`np.random.random()`, which produced
  meaningless slopes. See `src/gilg/evaluation/wefat.py`.
- **Secrets** come from environment variables, never hard-coded.
