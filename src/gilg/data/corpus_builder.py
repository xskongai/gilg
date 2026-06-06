"""Build the retrieval corpus as structured Documents.

The original pipeline exported the spreadsheets to a PDF, then re-extracted
text and split it by character count — which scrambled the original→inclusive
pairings. We skip that entirely: each pair/rule becomes one clean Document with
metadata, so retrieval returns intact mappings.
"""

from __future__ import annotations

from langchain_core.documents import Document

from gilg.data.loader import (
    load_counterfactual_pairs,
    load_cf_rules,
    load_gender_neutral_pairs,
)


def build_documents() -> list[Document]:
    docs: list[Document] = []

    # 1) Gender-neutral word pairs
    for p in load_gender_neutral_pairs():
        docs.append(
            Document(
                page_content=f"{p.original} -> {p.inclusive}",
                metadata={"source": "gender_neutral_pairs", "type": "word_pair"},
            )
        )

    # 2) Counterfactual rule strings
    for rule in load_cf_rules():
        docs.append(
            Document(
                page_content=rule,
                metadata={"source": "cf_rules", "type": "rule"},
            )
        )

    # 3) Counterfactual sentence pairs
    for s in load_counterfactual_pairs():
        docs.append(
            Document(
                page_content=f"{s.biased} -> {s.counterfactual}",
                metadata={
                    "source": "counterfactual_sentence_pairs",
                    "type": "sentence_pair",
                    "bias_type": s.bias_type,
                },
            )
        )

    return docs
