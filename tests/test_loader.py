"""Sanity checks on data loading and corpus building."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gilg.data.loader import (
    load_counterfactual_pairs,
    load_cf_rules,
    load_gender_neutral_pairs,
)
from gilg.data.corpus_builder import build_documents


def test_gender_neutral_pairs_load():
    pairs = load_gender_neutral_pairs()
    assert len(pairs) > 600
    assert all(p.original and p.inclusive for p in pairs)


def test_counterfactual_pairs_load():
    pairs = load_counterfactual_pairs()
    assert len(pairs) > 700


def test_cf_rules_load():
    rules = load_cf_rules()
    assert len(rules) > 100


def test_corpus_documents_have_metadata():
    docs = build_documents()
    assert len(docs) > 1500
    assert all("source" in d.metadata for d in docs)
