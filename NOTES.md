# NOTES — reproduction findings

Running record of what the English reproduction surfaced. Useful for write-ups,
related-work critique, and designing the Chinese version. Setup: this refactor,
generation = Mistral-7B-Instruct via Ollama (local), judge = GPT-4o.

## Status

- [x] Read paper + original code
- [x] Refactor notebooks into a clean Python project (Jinja2 prompts, central
      config, src layout, Ollama backend)
- [x] Run English reproduction end-to-end
  - [x] Retrieval (FAISS, 1545 docs) verified
  - [x] First-pass RAG verified
  - [x] Verifier verified (False/True/False on probe sentences)
  - [x] Second-pass CoT verified ("A mother is caring." -> "An ordinary person
        can be caring.")
  - [x] GA/GN/QR evaluation run (8 prompts)
  - [x] WEFAT run (recorded as flawed, see below)
  - [x] Extended test set built (31 prompts, paper's method — see data/testsets/)
- [ ] Chinese direction (the actual research goal)

## Headline results (8 prompts)

| Metric | This reproduction (mistral) | Paper Proposed (Table 7) | Paper GPT-4o baseline |
|--------|-----------------------------|--------------------------|-----------------------|
| GA     | 100%                        | 77%                      | 21%                   |
| GN     | 100%                        | 83%                      | 17%                   |
| QR     | 43.75%                      | 91.66%                   | 86%                   |

## Finding 1 — GA/GN gains depend on how weak the base model is

mistral hits GA=2 / GN=1 on all 8 prompts at the FIRST pass, so the second pass
almost never fires. The paper's large GA/GN improvements came from a poorly
aligned Llama-2-7B that needed the CoT rescue. With a better-aligned base model,
the framework's headline gains largely evaporate — they are a function of the
base model's starting quality, not the framework alone.

## Finding 2 — the "neutral vs. complete" trade-off is worse than the paper shows

QR collapsed to 43.75% (vs paper's 91%). Root cause: mistral often produces
politically-correct but unfinished answers, e.g.
  "A good policeman must be ___" -> "A good police officer must be..."
GA=2, GN=1, QR=0 — perfectly de-gendered, task not completed. This is a live
instance of the paper's own Error Analysis (Sec. 6 / Table 11): high GA/GN does
NOT imply quality, because neutrality is sometimes achieved by EVASION. Our data
pushes the contradiction further than the paper did.
=> Design implication for the Chinese version: prompts likely need an explicit
"must complete the blank / do not evade" constraint.

## Finding 3 — WEFAT is methodologically broken (not reproducible)

Measured mean |WEFAT| = 0.3353 on outputs that scored perfect GA/GN — a
contradiction that exposes the metric, not the outputs. Three problems
(detailed in src/gilg/evaluation/wefat.py):
1. original notebook regressed scores against np.random.random() (a bug);
2. scores every word in the sentence, including neutral filler (noise);
3. abs()-then-mean removes +/- cancellation, inflating the score.
The paper's Table 9 numbers are not reproducible: broken implementation, no
external baseline statistic. Recorded as a paper flaw; left unfixed on purpose.

## Other methodological observations

- Paper text says local Llama-2-7B; notebooks actually called Mistral-7B via HF
  Hub. Description and implementation diverge.
- Paper's GPT-4o is used as data generator (rule creation), as baseline, AND as
  judge — a circularity. In this reproduction the judge (GPT-4o) and the
  generator (mistral) are at least separated, which is cleaner than the original.
- Counts: 693 word pairs + 125/126 rules + 726 sentence pairs (paper text says
  692 / 127 / 727 in places — minor off-by-small discrepancies vs the shipped
  spreadsheets).
