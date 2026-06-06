"""Sample a Chinese test set from the project's own sentence pairs.

Takes 40 biased sentences (fixed seed) from data/raw_zh/counterfactual_sentence_pairs.xlsx
as evaluation queries.

CAVEAT: these sentences are ALSO in the retrieval corpus (not held out), so the
retriever can surface their gold rewrites directly. Scores on this set are
therefore OPTIMISTIC — they measure recall+application, not generalization. For
a generalization test, hold these IDs out of the index (not done here).

Outputs:
  data/testsets/queries_zh_40.txt   one biased sentence per line
  data/testsets/queries_zh_40.csv   with Pair ID + gold rewrite (traceable)
"""
import random
import pandas as pd
from pathlib import Path

df = pd.read_excel("data/raw_zh/counterfactual_sentence_pairs.xlsx")
random.seed(42)
test_ids = sorted(random.sample(list(df["Pair ID"]), 40))
test_df = df[df["Pair ID"].isin(test_ids)]

out = Path("data/testsets")
out.mkdir(parents=True, exist_ok=True)
with open(out / "queries_zh_40.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(test_df["Biased Sentence"].astype(str).str.strip()) + "\n")
test_df[["Pair ID", "Biased Sentence", "Counterfactual Sentence"]].to_csv(
    out / "queries_zh_40.csv", index=False
)
print(f"Wrote {len(test_df)} queries -> data/testsets/queries_zh_40.txt")
print("CAVEAT: overlaps with retrieval corpus; scores are optimistic.")
