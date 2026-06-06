"""Embedding model wrapper.

The whole pipeline uses ONE embedding model (set in config) so that the FAISS
index and every query live in the same vector space — fixing the original
MiniLM/mpnet dimension mismatch.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.config import cfg


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=cfg.retrieval.embedding_model)
