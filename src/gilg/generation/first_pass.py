"""First pass: retrieval-augmented generation of the initial response R1.

Retrieves top-k inclusive references and asks the LLM to answer the query
grounded in them. No CoT here — that's the second pass, triggered only if the
verifier flags R1 as non-inclusive.
"""

from __future__ import annotations

from dataclasses import dataclass

from gilg.generation.llm import get_llm
from gilg.retrieval.retriever import RetrievedChunk, Retriever
from gilg.utils.prompts import render
from config.config import cfg


@dataclass
class FirstPassResult:
    query: str
    response: str
    context: list[RetrievedChunk]


class FirstPass:
    def __init__(self, retriever: Retriever | None = None):
        self._retriever = retriever or Retriever()
        self._llm = get_llm()

    def run(self, query: str) -> FirstPassResult:
        chunks = self._retriever.retrieve(query)
        context = "\n".join(f"- {c.text}" for c in chunks)
        prompt = render(cfg.prompt.first_pass_template, context=context, question=query)
        response = str(self._llm.invoke(prompt)).strip()
        return FirstPassResult(query=query, response=response, context=chunks)
