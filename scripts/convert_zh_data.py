"""Convert the raw Chinese lexicon/sentence files into the English-aligned format.

Maps the uploaded Chinese columns to the same column names the English data uses,
so the existing loader reads them unchanged. Explanation columns (解释说明/改写说明)
are dropped for now to match the English schema; revisit later if needed.

Inputs (place in data/raw_zh/source/):
  cn_gi_lexicon_inclusive.xlsx   sheet cn_gi_lexicon       cols 编号/偏见词/改写词/解释说明
  cn_gi_sent_inclusive.xlsx      sheet gi_inclusive_rewrite cols 序号/例号/句子/性别包容改写/改写说明

Outputs (data/raw_zh/):
  gender_neutral_pairs.xlsx            Pair ID / Original Terms / Inclusive Terms
  counterfactual_sentence_pairs.xlsx   Pair ID / Biased Sentence / Counterfactual Sentence / Bias Type
"""
import sys
from pathlib import Path
import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw_zh/source")
OUT = Path("data/raw_zh")
OUT.mkdir(parents=True, exist_ok=True)

lex = pd.read_excel(SRC / "cn_gi_lexicon_inclusive.xlsx", sheet_name="cn_gi_lexicon")
lex_out = pd.DataFrame({
    "Pair ID": range(1, len(lex) + 1),
    "Original Terms": lex["偏见词"].astype(str).str.strip(),
    "Inclusive Terms": lex["改写词"].astype(str).str.strip(),
})
lex_out = lex_out[(lex_out["Original Terms"] != "") & (lex_out["Original Terms"] != "nan")]
lex_out.to_excel(OUT / "gender_neutral_pairs.xlsx", index=False)
print(f"词对表: {lex_out.shape} -> {OUT/'gender_neutral_pairs.xlsx'}")

sent = pd.read_excel(SRC / "cn_gi_sent_inclusive.xlsx", sheet_name="gi_inclusive_rewrite")

# Resolve "同N。" references: some rewrite cells just point to another row's
# rewrite (e.g. row 589 = "同266。" means "same rewrite as row 266"). Resolve
# them to the actual text, following chained references (e.g. 595 -> 356 -> 350).
import re

_REF = re.compile(r"^同\s*(\d+)。?\s*$")
_no_to_idx = {int(n): i for i, n in enumerate(sent["序号 / No."]) if pd.notna(n)}


def _resolve(no, seen=None):
    seen = seen or set()
    if no in seen or no not in _no_to_idx:
        return None
    seen.add(no)
    val = str(sent.iloc[_no_to_idx[no]]["性别包容改写"]).strip()
    m = _REF.match(val)
    return _resolve(int(m.group(1)), seen) if m else val


rewrites = []
for _, row in sent.iterrows():
    val = str(row["性别包容改写"]).strip()
    m = _REF.match(val)
    rewrites.append(_resolve(int(m.group(1))) or val if m else val)

sent_out = pd.DataFrame({
    "Pair ID": range(1, len(sent) + 1),
    "Biased Sentence": sent["句子 (CN data)"].astype(str).str.strip(),
    "Counterfactual Sentence": pd.Series(rewrites).astype(str).str.strip(),
    "Bias Type": "Gender",
})
sent_out = sent_out[(sent_out["Biased Sentence"] != "") & (sent_out["Biased Sentence"] != "nan")]
sent_out.to_excel(OUT / "counterfactual_sentence_pairs.xlsx", index=False)
print(f"句对表: {sent_out.shape} -> {OUT/'counterfactual_sentence_pairs.xlsx'}")
