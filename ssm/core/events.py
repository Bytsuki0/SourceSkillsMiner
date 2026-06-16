"""
Observer pattern for progress reporting (behavioural decoupling).

Previously every analyzer reported progress by calling ``print()`` directly,
hard-wiring the work to one output channel. Here the work (Subjects) only
*emits* :class:`ProgressEvent` objects; *what happens* with them is decided by
the attached :class:`Observer` implementations.

``ConsoleObserver`` reproduces the old console output. A different observer
(e.g. one that forwards structured ``stage``/``fraction`` to a web client)
can be attached without touching any analyzer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "ProgressEvent",
    "Observer",
    "ConsoleObserver",
    "Subject",
]


@dataclass
class ProgressEvent:
    """A single progress notification emitted by a Subject."""

    stage: str                               # short machine-friendly stage id
    message: str                             # human-readable line
    source: str = ""                         # name of the emitting Subject
    fraction: Optional[float] = None         # [0,1] progress within the stage, if known
    payload: Dict[str, object] = field(default_factory=dict)


class Observer(ABC):
    """Receives :class:`ProgressEvent` notifications from a Subject."""

    @abstractmethod
    def update(self, event: ProgressEvent) -> None:  # pragma: no cover - interface
        ...


class ConsoleObserver(Observer):
    """Prints events to stdout, preserving the original console experience."""

    def update(self, event: ProgressEvent) -> None:
        print(event.message)


class Subject:
    """
    Mixin that lets a class broadcast :class:`ProgressEvent` to observers.

    Observers attached to a parent (e.g. the facade) are propagated to the
    services/analyzers it builds, so a single ``attach()`` wires the whole run.
    """

    def __init__(self) -> None:
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> "Subject":
        if observer not in self._observers:
            self._observers.append(observer)
        return self

    def detach(self, observer: Observer) -> "Subject":
        if observer in self._observers:
            self._observers.remove(observer)
        return self

    def attach_all(self, observers: Optional[List[Observer]]) -> "Subject":
        for obs in observers or []:
            self.attach(obs)
        return self

    @property
    def observers(self) -> List[Observer]:
        return list(self._observers)

    def notify(self, event: ProgressEvent) -> None:
        for observer in self._observers:
            observer.update(event)

    def emit(
        self,
        message: str,
        *,
        stage: str = "",
        fraction: Optional[float] = None,
        **payload: object,
    ) -> None:
        """Convenience wrapper that builds and dispatches a ProgressEvent."""
        self.notify(
            ProgressEvent(
                stage=stage or self.__class__.__name__,
                message=message,
                source=self.__class__.__name__,
                fraction=fraction,
                payload=dict(payload),
            )
        )
