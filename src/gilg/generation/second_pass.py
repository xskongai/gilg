"""Second pass: CoT-guided rewrite of a non-inclusive R1 into R2.

Uses the externalized CoT prompt (prompts/<lang>/cot_rewrite.j2). The retrieved
context is rendered into the prompt; the LLM applies the step-by-step rewrite.
"""

from __future__ import annotations

from gilg.generation.first_pass import FirstPassResult
from gilg.generation.llm import get_llm, invoke_text
from config.config import cfg
from gilg.utils.prompts import render


class SecondPass:
    def __init__(self):
        self._llm = get_llm()

    def run(self, first: FirstPassResult) -> str:
        context = "\n".join(f"- {c.text}" for c in first.context)
        # The CoT prompt rewrites the model's response given the query + context.
        prompt = render(
            cfg.prompt.cot_template,
            context=context,
            question=first.query,
        )
        return invoke_text(self._llm, prompt)
