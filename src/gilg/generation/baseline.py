"""Zero-shot baseline: the same model answering the query as-is.

No retrieval, no verifier, no CoT — just the bare query fed to the model, which
is the paper's "as-is" baseline (Sec 5.2.1). Used for a clean ablation: same
model, baseline vs. full RAG+CoT pipeline, so the only variable is the framework.

Returns a PipelineResult shaped like Pipeline's, so the eval script treats both
uniformly (r1 == r_final, no second pass).
"""

from __future__ import annotations

from gilg.generation.llm import get_llm, invoke_text
from gilg.generation.pipeline import PipelineResult
from config.config import cfg
from gilg.utils.prompts import render


# Generation-task verbs mark a "description" prompt (open-ended writing);
# everything else (blanks to fill, bare statements) is "completion".
_DESCRIPTION_STARTS = ("write ", "suggest ", "describe ", "compose ", "create ")


def _classify(query: str) -> str:
    q = query.strip().lower()
    if q.startswith(_DESCRIPTION_STARTS):
        return "description"
    return "completion"


class Baseline:
    def __init__(self):
        self._llm = get_llm()

    def run(self, query: str) -> PipelineResult:
        qtype = _classify(query)
        prompt = render("baseline.j2", question=query, qtype=qtype)
        out = invoke_text(self._llm, prompt)
        return PipelineResult(
            query=query,
            r1=out,
            r_final=out,
            inclusive_after_first_pass=False,
            used_second_pass=False,
        )
