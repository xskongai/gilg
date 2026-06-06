"""Verifier: binary inclusive / non-inclusive check on R1.

The paper describes a Verifier that returns 1 (inclusive) or 0 (non-inclusive)
and triggers the second pass only on 0. There is no standalone verifier prompt
in the source; its definition matches the Gender Neutrality (GN) criterion
(0 = non-inclusive, 1 = inclusive), so we reuse that judge prompt here.
"""

from __future__ import annotations

import re

from config.config import cfg
from gilg.generation.llm import get_llm
from gilg.utils.prompts import render


def _parse_binary(text: str) -> int:
    m = re.search(r"[01]", text)
    return int(m.group()) if m else 0  # default to non-inclusive -> safer (runs pass 2)


class Verifier:
    def __init__(self):
        self._llm = get_llm()

    def is_inclusive(self, query: str, response: str) -> bool:
        prompt = render(cfg.prompt.judge_gn, prompt=query, response=response)
        out = str(self._llm.invoke(prompt))
        return _parse_binary(out) == 1
