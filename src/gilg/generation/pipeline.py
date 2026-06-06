"""Two-pass pipeline: R1 -> verifier -> (R2 only if non-inclusive) -> R_final.

This is the control flow the paper describes in its architecture figure but
that the original notebooks never assembled into one place. Branching logic
lives here, deliberately as plain Python rather than a LangChain chain, because
the flow is conditional, not linear.
"""

from __future__ import annotations

from dataclasses import dataclass

from gilg.generation.first_pass import FirstPass
from gilg.generation.second_pass import SecondPass
from gilg.generation.verifier import Verifier


@dataclass
class PipelineResult:
    query: str
    r1: str
    r_final: str
    inclusive_after_first_pass: bool
    used_second_pass: bool


class Pipeline:
    def __init__(self):
        self._first = FirstPass()
        self._verifier = Verifier()
        self._second = SecondPass()

    def run(self, query: str) -> PipelineResult:
        first = self._first.run(query)

        if self._verifier.is_inclusive(query, first.response):
            return PipelineResult(
                query=query,
                r1=first.response,
                r_final=first.response,
                inclusive_after_first_pass=True,
                used_second_pass=False,
            )

        r2 = self._second.run(first)
        return PipelineResult(
            query=query,
            r1=first.response,
            r_final=r2,
            inclusive_after_first_pass=False,
            used_second_pass=True,
        )
