"""Model registry — the single source of truth for which models exist and how
to reach each one.

Two explicit tables, cloud vs local, so routing is a lookup, never a guess:

  * LOCAL_MODELS  — run via local Ollama. Free, no API key. Just list the name.
  * CLOUD_MODELS  — run via an API. Each entry states its backend, base_url,
                    and the env var holding its key, plus optional quirks.

Why a registry instead of prefix matching:
  Model names across families collide ("qwen2.5" is local, "qwen3.7-max" is
  cloud; "glm4:9b" is local Ollama, cloud GLM lives elsewhere). Guessing by
  prefix (startswith("qwen") / startswith("glm")) routes the wrong ones to the
  wrong place. Exact lookup in these tables removes the ambiguity entirely.

To add a model: add ONE line to the correct table. Nothing else changes.
"""

from __future__ import annotations

# --- Local models: all run through Ollama. Name only. ------------------------
# Add a model here after `ollama pull <name>`.
LOCAL_MODELS: set[str] = {
    # English / general small models
    "mistral",
    "llama3.1",
    "gemma2:9b",
    # Chinese small models
    "qwen2.5",
    "glm4:9b",
    "yi:9b",
    # tiny variants (optional, for scaling experiments)
    "qwen2.5:0.5b",
    "llama3.2:1b",
    "gemma2:2b",
}

# --- Cloud models: each says exactly how to reach it. ------------------------
# backend ∈ {"openai", "gemini", "openai_compat"}
#   openai         -> official OpenAI endpoint
#   gemini         -> Google Gemini endpoint
#   openai_compat  -> any OpenAI-compatible endpoint (needs base_url)
# key_env: env var holding the API key.
# no_think (optional): True for reasoning models that default to thinking mode
#   and must be told to stop, or `content` comes back empty.
CLOUD_MODELS: dict[str, dict] = {
    "gpt-4o": {
        "backend": "openai",
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
    },
    "gemini-3.5-flash": {
        "backend": "gemini",
        "base_url": None,
        "key_env": "GOOGLE_API_KEY",
    },
    "deepseek-chat": {
        "backend": "openai_compat",
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "qwen3.7-max": {
        "backend": "openai_compat",
        "base_url": "https://ws-qda2js0k5wga6npk.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "key_env": "QWEN_API_KEY",
        "no_think": True,
    },
    # Cloud GLM (NOT the local glm4:9b — that one is in LOCAL_MODELS).
    "glm-4-plus": {
        "backend": "openai_compat",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "ZHIPU_API_KEY",
    },
}
