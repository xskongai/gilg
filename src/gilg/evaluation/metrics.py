"""Aggregation helpers for evaluation results.

Turns per-sample scores into the percentage summaries the paper reports
(GA/GN/QR as % of max), and computes simple human-vs-LLM agreement.
"""

from __future__ import annotations

from gilg.evaluation.judge import JudgeScores

# Max value per metric, used to normalize to a percentage.
_MAX = {"ga": 2, "gn": 1, "qr": 2}


def aggregate(scores: list[JudgeScores]) -> dict[str, float]:
    if not scores:
        return {"ga": 0.0, "gn": 0.0, "qr": 0.0}
    n = len(scores)
    return {
        "ga": 100 * sum(s.ga for s in scores) / (n * _MAX["ga"]),
        "gn": 100 * sum(s.gn for s in scores) / (n * _MAX["gn"]),
        "qr": 100 * sum(s.qr for s in scores) / (n * _MAX["qr"]),
    }


def agreement(a: list[int], b: list[int]) -> float:
    """Exact-match agreement (%) between two equal-length score lists."""
    if not a or len(a) != len(b):
        return 0.0
    return 100 * sum(int(x == y) for x, y in zip(a, b)) / len(a)
