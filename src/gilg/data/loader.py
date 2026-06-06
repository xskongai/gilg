"""Load the three source spreadsheets into plain Python records.

Columns are taken as-is from the files shipped with the paper:
  - gender_neutral_pairs.xlsx        : Pair ID, Original Terms, Inclusive Terms  (693)
  - cf_rules.xlsx                    : single column of "X: Y" rule strings        (125)
  - counterfactual_sentence_pairs.xlsx: Pair ID, Biased Sentence,
                                        Counterfactual Sentence, Bias Type         (726)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import config as _cfg


@dataclass
class WordPair:
    original: str
    inclusive: str


@dataclass
class SentencePair:
    biased: str
    counterfactual: str
    bias_type: str


def load_gender_neutral_pairs(path: Path | None = None) -> list[WordPair]:
    path = path or _cfg.GENDER_NEUTRAL_PAIRS
    df = pd.read_excel(path)
    return [
        WordPair(str(r["Original Terms"]).strip(), str(r["Inclusive Terms"]).strip())
        for _, r in df.iterrows()
        if pd.notna(r["Original Terms"]) and pd.notna(r["Inclusive Terms"])
    ]


def load_cf_rules(path: Path | None = None) -> list[str]:
    path = path or _cfg.CF_RULES
    if not Path(path).exists():
        return []  # zh has no cf_rules file; that's fine
    df = pd.read_excel(path)
    col = df.columns[0]  # the rule text lives in the (single) first column
    rules = [str(v).strip() for v in df[col] if pd.notna(v)]
    # the column header itself is also a rule line in this file
    header = str(col).strip()
    if header and ":" in header:
        rules = [header] + rules
    return rules


def load_counterfactual_pairs(path: Path | None = None) -> list[SentencePair]:
    path = path or _cfg.COUNTERFACTUAL_PAIRS
    df = pd.read_excel(path)
    return [
        SentencePair(
            str(r["Biased Sentence"]).strip(),
            str(r["Counterfactual Sentence"]).strip(),
            str(r.get("Bias Type", "")).strip(),
        )
        for _, r in df.iterrows()
        if pd.notna(r["Biased Sentence"]) and pd.notna(r["Counterfactual Sentence"])
    ]
