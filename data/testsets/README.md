# data/testsets/

Test prompts for evaluation, built by the paper's method.

## queries_extended (31 prompts)

Built by `scripts/build_testset.py`. Combines two sources (甲乙结合), labeled by
the paper's four bias types (Table 2):

| source        | count | what it is                                            |
|---------------|-------|-------------------------------------------------------|
| `crows_pairs` | 25    | real gender-category sentences from CrowS-Pairs        |
| `paper_table2`| 3     | verbatim anchors from the paper's Table 2              |
| `template`    | 3     | paper-style fill-ins for explicit_gender_marking, which CrowS-Pairs under-covers |

Bias-type coverage: explicit_gender_marking 5, gendered_pronoun 8,
stereotypical_bias 9, representational_bias 9.

The `.csv` keeps per-row `source` and `bias_type` for traceability; the `.txt`
is the plain one-query-per-line input for the eval scripts.

## Why not 100% from CrowS-Pairs

The paper says its prompts came from CrowS-Pairs / StereoSet / CoBIAS but never
published the actual list (a reproducibility gap). So this set reproduces the
paper's *construction method* with real CrowS-Pairs data as the body, rather
than claiming to be the paper's exact prompts. CrowS-Pairs also under-covers the
"explicit gender marking" category (occupational nouns like fireman/chairman),
so a few paper-style templates fill that gap — flagged as `source=template`.

## Rebuilding

```bash
curl -sL -o crows.csv https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv
python scripts/build_testset.py
```
