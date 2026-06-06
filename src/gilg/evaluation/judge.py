"""LLM-as-Judge: GA / GN / QR scoring.

Scores a (prompt, response) pair on the three paper metrics using GPT-4o and
the externalized judge prompts. Returns integers in the documented ranges:
GA 0-2, GN 0-1, QR 0-2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.config import cfg
from gilg.utils.prompts import render


@dataclass
class JudgeScores:
    ga: int  # gender assumption  (0-2)
    gn: int  # gender neutrality  (0-1)
    qr: int  # quality/relevance  (0-2)


def _extract_score(text: str, lo: int, hi: int) -> int:
    for tok in re.findall(r"-?\d+", text):
        v = int(tok)
        if lo <= v <= hi:
            return v
    return lo


class Judge:
    def __init__(self):
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.evaluation.judge_model
        self._temp = cfg.evaluation.judge_temperature

    def _ask(self, template: str, prompt: str, response: str) -> str:
        rendered = render(template, prompt=prompt, response=response)
        out = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temp,
            messages=[{"role": "user", "content": rendered}],
        )
        return out.choices[0].message.content or ""

    def score(self, prompt: str, response: str) -> JudgeScores:
        ga = _extract_score(self._ask(cfg.prompt.judge_ga, prompt, response), 0, 2)
        gn = _extract_score(self._ask(cfg.prompt.judge_gn, prompt, response), 0, 1)
        qr = _extract_score(self._ask(cfg.prompt.judge_qr, prompt, response), 0, 2)
        return JudgeScores(ga=ga, gn=gn, qr=qr)
