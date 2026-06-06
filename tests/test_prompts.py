"""Prompt rendering tests — verify externalized templates render with variables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gilg.utils.prompts import render
from config.config import cfg


def test_cot_renders_with_variables():
    out = render(cfg.prompt.cot_template, context="man -> person", question="A ___ is caring.")
    assert "man -> person" in out
    assert "A ___ is caring." in out
    assert "{{" not in out  # no unrendered placeholders


def test_judge_prompts_render():
    for tmpl in (cfg.prompt.judge_ga, cfg.prompt.judge_gn, cfg.prompt.judge_qr):
        out = render(tmpl, prompt="p", response="r")
        assert "p" in out and "r" in out
