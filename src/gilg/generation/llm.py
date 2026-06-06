"""LLM backend abstraction.

A single `get_llm()` returns a LangChain-compatible LLM. The backend
("ollama", "hf_hub", or "local") and all generation params come from config,
so swapping models never touches call sites.

Default: Mistral-7B-Instruct served locally via Ollama (free, runs on Apple
Silicon, reproducible). HF Hub now routes 7B+ models through paid third-party
providers, so it is no longer a free option.
"""

from __future__ import annotations

from functools import lru_cache

from config.config import cfg


@lru_cache(maxsize=1)
def get_llm():
    gen = cfg.generation

    if gen.backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=gen.gemini_model,
            google_api_key=cfg.google_api_key,
            temperature=gen.temperature,
            max_output_tokens=gen.max_new_tokens,
            top_p=gen.top_p,
        )

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
