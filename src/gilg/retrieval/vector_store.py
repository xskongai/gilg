"""FAISS vector store: build once, persist, reload.

Replaces the notebooks' repeated build-from-PDF logic. The index is created
from structured Documents and saved to disk; later runs just load it.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS

from config.config import cfg
from gilg.data.corpus_builder import build_documents
from gilg.retrieval.embedder import get_embedder
from gilg.utils.io import ensure_dir


def build_index(save: bool = True) -> FAISS:
    """Build a fresh FAISS index from the source corpus."""
    docs = build_documents()
    store = FAISS.from_documents(docs, get_embedder())
    if save:
        ensure_dir(cfg.retrieval.index_path.parent)
        store.save_local(str(cfg.retrieval.index_path))
    return store


def load_index(path: Path | None = None) -> FAISS:
    """Load a persisted FAISS index."""
    path = path or cfg.retrieval.index_path
    return FAISS.load_local(
        str(path),
        get_embedder(),
        allow_dangerous_deserialization=True,  # our own locally-built index
    )


def get_index() -> FAISS:
    """Load the index if present, otherwise build (and save) it."""
    if cfg.retrieval.index_path.exists():
        return load_index()
    return build_index(save=True)
