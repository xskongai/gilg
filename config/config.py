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
    compat_base_url: str | None = None  # set by set_model preset or env

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

    Works for both large cloud models and small local models — you give the
    model name and the backend is inferred:

      Large / cloud (need an API key in the environment):
        set_model("gpt-4o")            -> openai        (OPENAI_API_KEY)
        set_model("gpt-4o-mini")       -> openai
        set_model("o3-mini")           -> openai
        set_model("gemini-2.5-flash")  -> gemini        (GOOGLE_API_KEY)
        set_model("deepseek-chat")     -> openai_compat (DEEPSEEK_API_KEY)
        set_model("deepseek-reasoner") -> openai_compat
        set_model("qwen-max")          -> openai_compat (DASHSCOPE_API_KEY)
        set_model("moonshot-v1-8k")    -> openai_compat (MOONSHOT_API_KEY)

      Small / local (free, no key):
        set_model("mistral")           -> ollama
        set_model("qwen2.5")           -> ollama
        set_model("qwen2.5:0.5b")      -> ollama  (tiny model)
        set_model("llama3.2:1b")       -> ollama  (tiny model)
        set_model("gemma2:2b")         -> ollama  (small model)
        set_model("phi3")              -> ollama

    Escape hatches for anything not in the table:
        set_model("openai:gpt-4.1")            force OpenAI
        set_model("gemini:gemini-2.5-pro")     force Gemini
        set_model("ollama:my-custom-model")    force Ollama
        set_model("compat:my-model@https://host/v1")  arbitrary OpenAI-compatible

    Must be called before the LLM is first built; clears the cached LLM.
    """
    backend, model, base_url, key_env = _resolve_model(name)

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
        # Resolve the key from the provider-specific env var if present,
        # else COMPAT_API_KEY, else leave whatever is already set.
        if key_env:
            cfg.compat_api_key = (
                os.getenv(key_env)
                or os.getenv("COMPAT_API_KEY")
                or cfg.compat_api_key
            )

    _clear_caches()


# Provider presets for OpenAI-compatible endpoints: model-name prefix ->
# (base_url, api-key env var). Add a line here to support a new provider.
_COMPAT_PRESETS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    # Cloud Qwen (DashScope) uses hyphenated names: qwen-max, qwen-plus, ...
    # Note: local Ollama Qwen is "qwen2.5" / "qwen2.5:0.5b" and is matched by
    # the prefix below only via the hyphen, so it correctly falls through to
    # Ollama. Use set_model("compat:qwen2.5-...") to force the cloud variant.
    # "qwen-": ("https://ws-qda2js0k5wga6npk.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "QWEN_API_KEY"),
    "qwen3": ("https://ws-qda2js0k5wga6npk.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "QWEN_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
}


def _resolve_model(name: str):
    """Map a model name to (backend, model, base_url, key_env).

    Order: explicit "backend:..." prefix > known cloud families > compat
    presets > default to local Ollama.
    """
    # 1) Explicit prefix escape hatch.
    if ":" in name and name.split(":", 1)[0] in {
        "openai", "gemini", "ollama", "hf", "compat"
    }:
        kind, rest = name.split(":", 1)
        if kind == "openai":
            return "openai", rest, None, None
        if kind == "gemini":
            return "gemini", rest, None, None
        if kind == "ollama":
            return "ollama", rest, None, None
        if kind == "compat":
            # "compat:model@https://host/v1"
            if "@" in rest:
                m, url = rest.split("@", 1)
                return "openai_compat", m, url, None
            return "openai_compat", rest, cfg.generation.compat_base_url, None

    low = name.lower()

    # 2) Known cloud families.
    if low.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai", name, None, None
    if low.startswith("gemini"):
        return "gemini", name, None, None

    # 3) OpenAI-compatible providers by name prefix.
    for prefix, (url, key_env) in _COMPAT_PRESETS.items():
        if low.startswith(prefix):
            return "openai_compat", name, url, key_env

    # 4) Default: treat as a local Ollama model (covers all small models,
    #    e.g. qwen2.5:0.5b, llama3.2:1b, gemma2:2b, phi3, ...).
    return "ollama", name, None, None


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
