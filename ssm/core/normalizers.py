"""
Numeric normalizers used by the scorers.

Consolidates the helpers that were previously copy-pasted across ScoringSys
(``_safe_log_norm``, ``_linear_cap``, ``to_01``, ``_to_signed01``) and the
analyzers (``_clamp01``, ``_to_signed``). Behaviour is preserved exactly.
"""

import math

__all__ = ["safe_log_norm", "linear_cap", "clamp01", "to_01", "to_signed"]


def safe_log_norm(x: float, scale: float = 100.0) -> float:
    """
    Logarithmic normalization mapping x in [0, +inf) to [0, 1].
    Returns 0.0 for any non-numeric input so API failures degrade gracefully.
    """
    if not isinstance(x, (int, float)):
        return 0.0
    if x <= 0:
        return 0.0
    try:
        return min(1.0, math.log10(x + 1) / math.log10(scale + 1))
    except Exception:
        return 0.0


def linear_cap(x: float, cap: float) -> float:
    """Map x to [0, 1] linearly, saturating at ``cap``."""
    if not isinstance(x, (int, float)):
        return 0.0
    if x <= 0:
        return 0.0
    return min(1.0, x / cap)


def clamp01(x: float) -> float:
    """Clamp x into the closed interval [0, 1]."""
    return max(0.0, min(1.0, x))


def to_01(score: float) -> float:
    """Map a signed [-1, 1] score into [0, 1]."""
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def to_signed(x01: float) -> float:
    """Map a [0, 1] score into [-1, 1]."""
    return max(-1.0, min(1.0, 2.0 * x01 - 1.0))
