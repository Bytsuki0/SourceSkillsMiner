"""
Configuration loading.

Replaces the per-module ``configparser`` blocks (one in nearly every old god
file) with a single :class:`Config` value object. Reading is relative to the
current working directory, exactly as before — the Flask backend sets the cwd
per job and drops a ``config.ini`` there.
"""

from __future__ import annotations

import configparser as cfgparser
from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["Config", "DEFAULT_WEIGHTS"]

# Scoring area weights. Kept identical to the previous ScoringSys defaults.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "OSS": 1.0,
    "Status": 1.0,
    "Adaptability": 1.0,
    "Sentiment": 1.0,
    "Commitment": 1.0,
}


@dataclass
class Config:
    """Resolved GitHub credentials and scoring weights."""

    username: Optional[str] = None
    token: Optional[str] = None
    weights: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = DEFAULT_WEIGHTS.copy()

    @classmethod
    def from_file(cls, path: str = "config.ini") -> "Config":
        """Load credentials + weights from an ``config.ini`` file."""
        parser = cfgparser.ConfigParser()
        parser.read(path)
        username = parser.get("github", "username", fallback=None)
        token = parser.get("github", "token", fallback=None)
        return cls(
            username=username,
            token=token,
            weights=cls._load_weights(parser),
        )

    @staticmethod
    def _load_weights(parser: cfgparser.ConfigParser) -> Dict[str, float]:
        if "scoring" not in parser:
            return DEFAULT_WEIGHTS.copy()
        section = parser["scoring"]
        weights: Dict[str, float] = {}
        for key, default in DEFAULT_WEIGHTS.items():
            try:
                weights[key] = float(section.get(key, default))
            except Exception:
                weights[key] = default
        return weights

    def normalized_weights(
        self, overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Return the weights scaled so their absolute values sum to 1.

        Mirrors the normalization that used to live inline in ScoringSys.score_user.
        """
        weights = self.weights.copy()
        if overrides:
            weights.update(overrides)

        total = sum(abs(v) for v in weights.values())
        if total == 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: float(v) / total for k, v in weights.items()}
