"""LLM backend abstraction.

A single `get_llm()` returns a LangChain-compatible LLM. The backend and all
generation params come from config, so swapping models never touches call
sites (first_pass / second_pass / verifier / pipeline stay unchanged).

Supported backends
-------------------
Large / cloud models (API key required):
  - "openai"          OpenAI GPT (gpt-4o, gpt-4o-mini, o1, o3-mini, ...)
  - "gemini"          Google Gemini (gemini-2.5-flash, gemini-2.5-pro, ...)
  - "openai_compat"   ANY OpenAI-compatible endpoint via base_url:
                      DeepSeek, Qwen (DashScope/maas), Moonshot, Together,
                      Groq, vLLM/SGLang servers, etc.

Small / local models (free, reproducible, run on a laptop):
  - "ollama"          local Ollama (mistral, qwen2.5, qwen2.5:0.5b,
                      llama3.2:1b, gemma2:2b, phi3, ...)
  - "hf_hub"          HuggingFace Inference Endpoint
  - "local"           in-process transformers pipeline (fully offline)

Default: Mistral-7B via Ollama. You normally don't set the backend by hand —
`set_model("gpt-4o")`, `set_model("deepseek-chat")`, `set_model("qwen2.5")`
resolve it via config/model_registry.py (see config.set_model).
"""

from __future__ import annotations

from functools import lru_cache

from config.config import cfg


@lru_cache(maxsize=1)
def get_llm():
    gen = cfg.generation

    # ---- Large / cloud models -------------------------------------------

    if gen.backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=gen.openai_model,
            api_key=cfg.openai_api_key,
            temperature=gen.temperature,
            max_tokens=gen.max_new_tokens,
            top_p=gen.top_p,
        )

    if gen.backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=gen.gemini_model,
            google_api_key=cfg.google_api_key,
            temperature=gen.temperature,
            max_output_tokens=gen.max_new_tokens,
            top_p=gen.top_p,
        )

    if gen.backend == "openai_compat":
        # Any OpenAI-compatible API: DeepSeek, Qwen/DashScope/maas, Moonshot,
        # Together, Groq, local vLLM/SGLang, ... Distinguished only by
        # base_url + api_key, so one branch covers them all.
        from langchain_openai import ChatOpenAI

        kwargs = dict(
            model=gen.compat_model,
            api_key=cfg.compat_api_key,
            base_url=gen.compat_base_url,
            temperature=gen.temperature,
            max_tokens=gen.max_new_tokens,
            top_p=gen.top_p,
        )
        # Some cloud reasoning models (e.g. Qwen3 family) default to "thinking"
        # mode, which puts the chain-of-thought in `reasoning_content` and can
        # leave the final `content` empty (so invoke_text returns ""). The
        # registry marks such models with no_think=True; disable thinking via
        # the common knobs (unknown keys are ignored by the server).
        if gen.compat_no_think:
            kwargs["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        return ChatOpenAI(**kwargs)

    # ---- Small / local models -------------------------------------------

    if gen.backend == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=gen.ollama_model,
            temperature=gen.temperature,
            num_predict=gen.max_new_tokens,
            top_p=gen.top_p,
        )

    if gen.backend == "hf_hub":
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=gen.repo_id,
            huggingfacehub_api_token=cfg.hf_token,
            temperature=gen.temperature,
            max_new_tokens=gen.max_new_tokens,
            top_p=gen.top_p,
        )

    if gen.backend == "local":
        from langchain_huggingface import HuggingFacePipeline

        return HuggingFacePipeline.from_model_id(
            model_id=gen.repo_id,
            task="text-generation",
            pipeline_kwargs={
                "temperature": gen.temperature,
                "max_new_tokens": gen.max_new_tokens,
                "top_p": gen.top_p,
            },
        )

    raise ValueError(f"Unknown generation backend: {gen.backend!r}")


def invoke_text(llm, prompt: str) -> str:
    """Call an LLM and return its response as a clean string.

    Backends differ in return type: Ollama / HF pipelines return a plain
    string, while chat models (ChatOpenAI, Gemini, OpenAI-compatible) return
    a LangChain message object whose text lives in `.content`. This helper
    normalises both so call sites never see metadata (`additional_kwargs`,
    `response_metadata`, ...) leak into outputs.
    """
    out = llm.invoke(prompt)
    content = getattr(out, "content", out)  # AIMessage -> .content; str -> itself
    if isinstance(content, list):
        # Some chat models return a list of content blocks.
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip()