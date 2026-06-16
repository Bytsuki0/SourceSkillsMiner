"""
Common base for all analyzers.

Each analyzer is a :class:`~ssm.core.events.Subject` so it can broadcast
progress through the Observer chain instead of calling ``print()`` directly.
The facade attaches observers and shares the HTTP client / services with every
analyzer it builds, keeping the analyzers decoupled from how data is fetched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ssm.core.client import GitHubClient
from ssm.core.events import Subject

__all__ = ["Analyzer"]


class Analyzer(Subject, ABC):
    """Base class: an observable unit of analysis bound to a GitHub user."""

    def __init__(self, client: GitHubClient, username: str) -> None:
        Subject.__init__(self)
        self.client = client
        self.username = username

    @abstractmethod
    def analyze(self) -> dict:  # pragma: no cover - interface
        """Run the analysis and return a result dict."""
        ...
