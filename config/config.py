"""Central configuration — the single source of truth for the whole project.

Every model name, path, and hyperparameter lives here so nothing is hard-coded
across modules. Values mirror what the original notebooks actually ran with.
Override any field via environment variables (see `.env.example`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Paths

ROOT = Path(__file__).resolve().parent.parent

# Load a local .env (if present) so secrets like HF_TOKEN / OPENAI_API_KEY are
# available via os.getenv without exporting them by hand. No-op if absent.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
PROMPTS_DIR = ROOT / "prompts"

# Raw source files (the three spreadsheets shipped with the paper)
GENDER_NEUTRAL_PAIRS = RAW_DIR / "gender_neutral_pairs.xlsx"  # 693 word pairs
CF_RULES = RAW_DIR / "cf_rules.xlsx"  # 125 rules
COUNTERFACTUAL_PAIRS = RAW_DIR / "counterfactual_sentence_pairs.xlsx"  # 726 sentence pairs


# Retrieval / embedding

@dataclass
class RetrievalConfig:
    # IMPORTANT: the original notebooks mixed all-MiniLM-L6-v2 (384-d) and
    # all-mpnet-base-v2 (768-d) across passes, producing incompatible indexes.
    # We pick ONE model for the entire pipeline. mpnet is the stronger default.
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    top_k: int = 5  # paper: k = 5
    chunk_size: int = 500  # paper: RecursiveCharacterTextSplitter
    chunk_overlap: int = 50
    index_path: Path = INDEX_DIR / "en" / "faiss_index"


# Generation

@dataclass
class GenerationConfig:
    # Default: Mistral-7B-Instruct via Ollama (local, free, reproducible).
    # The paper text says local Llama-2-7B and the notebooks called Mistral-7B
    # via HF Hub; HF no longer serves 7B models for free, so we run it locally.
    #
    # backend ∈ {"openai", "gemini", "openai_compat",   # large / cloud
    #            "ollama", "hf_hub", "local"}            # small / local
    # You rarely set this by hand: set_model() infers it from the model name.
    backend: str = "ollama"

    # Per-backend model names (only the active backend's field is read).
    ollama_model: str = "mistral"  # `ollama pull mistral`
    gemini_model: str = "gemini-2.5-flash"  # backend == "gemini"
    openai_model: str = "gpt-4o"  # backend == "openai"
    repo_id: str = "mistralai/Mistral-7B-Instruct-v0.1"  # hf_hub / local

    # OpenAI-compatible endpoints (DeepSeek, Qwen/DashScope, Moonshot, ...).
    compat_model: str = "deepseek-chat"
    compat_base_url: str | None = None  # set by set_model from the registry
    compat_no_think: bool = False  # True for cloud reasoning models (disable thinking)

    temperature: float = 0.7  # paper hyperparameters
    max_new_tokens: int = 512
    top_p: float = 0.9


# Evaluation (LLM-as-Judge)

@dataclass
class EvaluationConfig:
    judge_model: str = "gpt-4o"  # paper used GPT-4o as judge
    judge_temperature: float = 0.0  # deterministic judging
    glove_path: Path = DATA_DIR / "glove.6B.100d.txt"  # for WEFAT


# Prompts / language switching

@dataclass
class PromptConfig:
    lang: str = "en"  # switch to "zh" later; templates parallel
    first_pass_template: str = "first_pass.j2"
    cot_template: str = "cot_rewrite.j2"
    judge_ga: str = "judge/gender_assumption.j2"
    judge_gn: str = "judge/gender_neutrality.j2"
    judge_qr: str = "judge/quality_relevance.j2"


# Language profiles — one bundle of language-bound settings per language.
# Switching language (set_lang) swaps data dir, embedding model, default
# generation model, and index subdir together. English values are unchanged.

LANG_PROFILES = {
    "en": {
        "data_dir": RAW_DIR,  # data/raw
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "ollama_model": "mistral",
        "index_subdir": "en",
    },
    "zh": {
        "data_dir": DATA_DIR / "raw_zh",  # data/raw_zh
        "embedding_model": "BAAI/bge-base-zh-v1.5",
        "ollama_model": "qwen2.5",
        "index_subdir": "zh",
    },
}


# Top-level config

@dataclass
class Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    seed: int = 42

    # Secrets pulled from the environment, never hard-coded.
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    # Key for OpenAI-compatible endpoints (DeepSeek/Qwen/...). Falls back to a
    # provider-specific env var (e.g. DEEPSEEK_API_KEY) set by set_model().
    compat_api_key: str | None = field(default_factory=lambda: os.getenv("COMPAT_API_KEY"))


# A ready-to-import default instance.
cfg = Config()


def active_model_name() -> str:
    """Return the name of the generation model currently in effect.

    Single source of truth for "which model am I actually running" across all
    backends — used for run-dir naming and the summary CSV so per-model runs
    are correctly labelled.
    """
    b = cfg.generation.backend
    return {
        "openai": cfg.generation.openai_model,
        "gemini": cfg.generation.gemini_model,
        "openai_compat": cfg.generation.compat_model,
        "ollama": cfg.generation.ollama_model,
        "hf_hub": cfg.generation.repo_id,
        "local": cfg.generation.repo_id,
    }.get(b, b)


def set_lang(lang: str) -> None:
    """Switch the active language and all language-bound settings at once.

    Applies the LANG_PROFILES bundle for `lang`: data directory, embedding
    model, default generation model, FAISS index subdir, and prompt language.
    Also rebinds the module-level data-file paths the loader reads, and clears
    cached LLM/embedder/index so the new language takes effect in-process.
    """
    if lang not in LANG_PROFILES:
        raise ValueError(f"Unknown lang {lang!r}; choose from {list(LANG_PROFILES)}")
    p = LANG_PROFILES[lang]

    cfg.prompt.lang = lang
    cfg.retrieval.embedding_model = p["embedding_model"]
    cfg.retrieval.index_path = INDEX_DIR / p["index_subdir"] / "faiss_index"
    cfg.generation.ollama_model = p["ollama_model"]

    # Rebind the data-file paths used by the loader.
    global GENDER_NEUTRAL_PAIRS, CF_RULES, COUNTERFACTUAL_PAIRS, RAW_DIR
    RAW_DIR = p["data_dir"]
    GENDER_NEUTRAL_PAIRS = RAW_DIR / "gender_neutral_pairs.xlsx"
    CF_RULES = RAW_DIR / "cf_rules.xlsx"
    COUNTERFACTUAL_PAIRS = RAW_DIR / "counterfactual_sentence_pairs.xlsx"

    _clear_caches()


def set_model(name: str) -> None:
    """Override the generation model at runtime (e.g. from a --model CLI arg).

    Routing is a lookup in config/model_registry.py, never a prefix guess:

      1) explicit "backend:..." escape hatch (openai/gemini/ollama/compat)
      2) exact match in CLOUD_MODELS  -> use its backend/base_url/key_env
      3) exact match in LOCAL_MODELS  -> Ollama
      4) otherwise -> treat as a local Ollama model (with a note)

    Examples:
      set_model("gpt-4o")        -> openai        (in CLOUD_MODELS)
      set_model("qwen3.7-max")   -> openai_compat (in CLOUD_MODELS, no_think)
      set_model("qwen2.5")       -> ollama        (in LOCAL_MODELS)
      set_model("glm4:9b")       -> ollama        (in LOCAL_MODELS, NOT cloud GLM)

    Escape hatches for anything not in the registry:
      set_model("openai:gpt-4.1")
      set_model("gemini:gemini-2.5-pro")
      set_model("ollama:my-custom-model")
      set_model("compat:my-model@https://host/v1")

    Must be called before the LLM is first built; clears the cached LLM.
    """
    backend, model, base_url, key_env, no_think = _resolve_model(name)

    cfg.generation.backend = backend
    if backend == "openai":
        cfg.generation.openai_model = model
    elif backend == "gemini":
        cfg.generation.gemini_model = model
    elif backend == "ollama":
        cfg.generation.ollama_model = model
    elif backend == "openai_compat":
        cfg.generation.compat_model = model
        cfg.generation.compat_base_url = base_url
        cfg.generation.compat_no_think = no_think
        # Resolve the key from the provider-specific env var if present,
        # else COMPAT_API_KEY, else leave whatever is already set.
        if key_env:
            cfg.compat_api_key = (
                os.getenv(key_env)
                or os.getenv("COMPAT_API_KEY")
                or cfg.compat_api_key
            )

    _clear_caches()


def _resolve_model(name: str):
    """Map a model name to (backend, model, base_url, key_env, no_think).

    Lookup order: explicit prefix > CLOUD_MODELS > LOCAL_MODELS > Ollama fallback.
    Exact-match lookup (not prefix matching) means same-family names that live
    in different places — e.g. local "qwen2.5" vs cloud "qwen3.7-max", or local
    "glm4:9b" vs cloud GLM — are never confused.
    """
    from config.model_registry import CLOUD_MODELS, LOCAL_MODELS

    # 1) Explicit "backend:..." escape hatch (always wins).
    if ":" in name and name.split(":", 1)[0] in {"openai", "gemini", "ollama", "hf", "compat"}:
        kind, rest = name.split(":", 1)
        if kind == "openai":
            return "openai", rest, None, None, False
        if kind == "gemini":
            return "gemini", rest, None, None, False
        if kind == "ollama":
            return "ollama", rest, None, None, False
        if kind == "compat":
            # "compat:model@https://host/v1"
            if "@" in rest:
                m, url = rest.split("@", 1)
                return "openai_compat", m, url, None, False
            return "openai_compat", rest, cfg.generation.compat_base_url, None, False

    # 2) Exact match in the cloud registry.
    if name in CLOUD_MODELS:
        spec = CLOUD_MODELS[name]
        return (
            spec["backend"],
            name,
            spec.get("base_url"),
            spec.get("key_env"),
            spec.get("no_think", False),
        )

    # 3) Exact match in the local registry -> Ollama.
    if name in LOCAL_MODELS:
        return "ollama", name, None, None, False

    # 4) Fallback: unknown name, assume a local Ollama model.
    #    (Covers ad-hoc pulls not yet added to LOCAL_MODELS.)
    print(f"[config] '{name}' not in registry; assuming local Ollama model. "
          f"Add it to LOCAL_MODELS or CLOUD_MODELS to be explicit.")
    return "ollama", name, None, None, False


def set_temperature(value: float) -> None:
    """Override generation temperature at runtime (e.g. --temperature 0).

    Use 0 for deterministic, reproducible runs; 0.7 matches the paper. Must be
    called before the LLM is first built; clears the cached LLM.
    """
    cfg.generation.temperature = value
    _clear_llm_cache()


def _clear_caches() -> None:
    """Clear cached LLM and embedder so model/lang changes take effect."""
    try:
        from gilg.generation.llm import get_llm

        get_llm.cache_clear()
    except Exception:
        pass
    try:
        from gilg.retrieval.embedder import get_embedder

        get_embedder.cache_clear()
    except Exception:
        pass


def _clear_llm_cache() -> None:  # kept for backward compat
    _clear_caches()