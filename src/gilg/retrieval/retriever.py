"""Top-k retrieval over the FAISS index."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS

from config.config import cfg
from gilg.retrieval.vector_store import get_index


@dataclass
class RetrievedChunk:
    text: str
    source: str


class Retriever:
    def __init__(self, index: FAISS | None = None, k: int | None = None):
        self._index = index or get_index()
        self._k = k or cfg.retrieval.top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        docs = self._index.similarity_search(query, k=self._k)
        return [
            RetrievedChunk(d.page_content, d.metadata.get("source", "unknown"))
            for d in docs
        ]
