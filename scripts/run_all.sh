#!/usr/bin/env bash
# =============================================================================
# run_all.sh — batch-run the GILG evaluation across models and languages.
#
# Runs each model twice: BASELINE (bare model) and PROPOSED (full RAG+CoT).
# Organized into four groups: EN-big / EN-small / ZH-big / ZH-small.
#
# Usage:
#   bash scripts/run_all.sh              # run everything enabled below
#   RUN_EN_SMALL=1 RUN_REST=0 bash scripts/run_all.sh   # only EN small models
#
# Edit the CONFIG block to change filenames, models, or which groups run.
# =============================================================================

set -u  # error on undefined variables (but keep going on model failures)

# ----------------------------- CONFIG ----------------------------------------
# --- test set filenames (single source of truth; change here only) ---
EN_FILE="data/testsets/queries_en_paper20.txt"
ZH_FILE="data/testsets/queries_zh_paper20.txt"

# --- generation temperature (0 = deterministic / reproducible) ---
TEMP=0

# --- model lists (space-separated). Edit to add/remove models. ---
# Big / cloud models (need API keys in .env).
EN_BIG_MODELS="gpt-4o gemini-3.5-flash deepseek-chat"
ZH_BIG_MODELS="gpt-4o gemini-3.5-flash deepseek-chat qwen3.7-max"

# Small / local Ollama models (free, no key besides the gpt-4o judge).
EN_SMALL_MODELS="mistral llama3.1 gemma2:9b"
ZH_SMALL_MODELS="qwen2.5 glm4:9b yi:9b"

# --- group switches: set to 1 to run, 0 to skip ---
RUN_EN_BIG=${RUN_EN_BIG:-0}      # off by default (costs API money)
RUN_EN_SMALL=${RUN_EN_SMALL:-1}
RUN_ZH_BIG=${RUN_ZH_BIG:-0}      # off by default (costs API money)
RUN_ZH_SMALL=${RUN_ZH_SMALL:-1}

# convenience: RUN_REST toggles the big-model groups together
if [ "${RUN_REST:-}" = "0" ]; then RUN_EN_BIG=0; RUN_ZH_BIG=0; fi
if [ "${RUN_REST:-}" = "1" ]; then RUN_EN_BIG=1; RUN_ZH_BIG=1; fi
# -----------------------------------------------------------------------------

SCRIPT="scripts/run_evaluation.py"

# run one model: both baseline and proposed
run_one() {
  local file="$1" lang="$2" model="$3"
  echo ""
  echo "================================================================"
  echo ">>> [$lang] $model  —  BASELINE"
  echo "================================================================"
  python "$SCRIPT" --file "$file" --lang "$lang" --model "$model" --baseline --temperature "$TEMP" \
    || echo "!!! FAILED: [$lang] $model baseline (continuing)"

  echo ""
  echo "================================================================"
  echo ">>> [$lang] $model  —  PROPOSED (RAG+CoT)"
  echo "================================================================"
  python "$SCRIPT" --file "$file" --lang "$lang" --model "$model" --temperature "$TEMP" \
    || echo "!!! FAILED: [$lang] $model proposed (continuing)"
}

# run a whole group
run_group() {
  local file="$1" lang="$2" label="$3"; shift 3
  local models="$*"
  echo ""
  echo "################################################################"
  echo "#  GROUP: $label   file=$file"
  echo "########################################################"
  for m in $models; do
    run_one "$file" "$lang" "$m"
  done
}

# ----------------------------- DISPATCH --------------------------------------
echo "run_all.sh starting"
echo "  EN_FILE = $EN_FILE"
echo "  ZH_FILE = $ZH_FILE"
echo "  groups: EN_BIG=$RUN_EN_BIG EN_SMALL=$RUN_EN_SMALL ZH_BIG=$RUN_ZH_BIG ZH_SMALL=$RUN_ZH_SMALL"

[ "$RUN_EN_SMALL" = "1" ] && run_group "$EN_FILE" en "EN / small (local)"  $EN_SMALL_MODELS
[ "$RUN_EN_BIG"   = "1" ] && run_group "$EN_FILE" en "EN / big (cloud)"    $EN_BIG_MODELS
[ "$RUN_ZH_SMALL" = "1" ] && run_group "$ZH_FILE" zh "ZH / small (local)"  $ZH_SMALL_MODELS
[ "$RUN_ZH_BIG"   = "1" ] && run_group "$ZH_FILE" zh "ZH / big (cloud)"    $ZH_BIG_MODELS

echo ""
echo "run_all.sh done. Results in results/runs/<timestamp>_<model>[_baseline]/"
