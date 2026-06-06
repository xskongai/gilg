"""Prompt rendering via Jinja2.

All prompts live as `.j2` files under `prompts/<lang>/`. Code never embeds
prompt text; it calls `render()`. Switching language is a single argument,
which keeps English and (future) Chinese prompts fully decoupled from logic.
"""

from __future__ import annotations

from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from config.config import cfg, PROMPTS_DIR


@lru_cache(maxsize=None)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        # Keep `{% for %}` blocks from leaving blank lines in the prompt text.
        trim_blocks=True,
        lstrip_blocks=True,
        # Prompts are plain text, not HTML — never escape quotes/angle brackets.
        autoescape=select_autoescape(enabled_extensions=()),
        # Fail loudly if a template variable is missing instead of silently "".
        undefined=StrictUndefined,
    )


def render(template_name: str, lang: str | None = None, **kwargs) -> str:
    """Render a prompt template.

    Args:
        template_name: path relative to a language dir, e.g. "cot_rewrite.j2"
                       or "judge/gender_assumption.j2".
        lang: language subfolder; defaults to cfg.prompt.lang ("en").
        **kwargs: template variables (e.g. context=..., question=...).
    """
    lang = lang or cfg.prompt.lang
    template = _env().get_template(f"{lang}/{template_name}")
    return template.render(**kwargs)
