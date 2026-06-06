"""Pipeline wiring test with stubbed components (no network/model calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gilg.generation.pipeline import Pipeline
from gilg.generation.first_pass import FirstPassResult
from gilg.retrieval.retriever import RetrievedChunk


def test_second_pass_skipped_when_inclusive(monkeypatch):
    p = Pipeline.__new__(Pipeline)
    p._first = type("F", (), {"run": lambda self, q: FirstPassResult(q, "inclusive R1", [RetrievedChunk("x", "s")])})()
    p._verifier = type("V", (), {"is_inclusive": lambda self, q, r: True})()
    p._second = type("S", (), {"run": lambda self, f: "R2"})()
    res = p.run("q")
    assert res.used_second_pass is False
    assert res.r_final == "inclusive R1"


def test_second_pass_runs_when_non_inclusive():
    p = Pipeline.__new__(Pipeline)
    p._first = type("F", (), {"run": lambda self, q: FirstPassResult(q, "biased R1", [RetrievedChunk("x", "s")])})()
    p._verifier = type("V", (), {"is_inclusive": lambda self, q, r: False})()
    p._second = type("S", (), {"run": lambda self, f: "R2 inclusive"})()
    res = p.run("q")
    assert res.used_second_pass is True
    assert res.r_final == "R2 inclusive"
