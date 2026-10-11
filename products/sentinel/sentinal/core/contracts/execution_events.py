from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentinal.core.event import Event


# --------------------------------------------------------------------
# Execution lifecycle
# --------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ExecutionStarted(Event[None]):
    """
    Published immediately before execution begins.
    """


@dataclass(slots=True, frozen=True)
class ExecutionCompleted(Event[Any]):
    """
    Published when execution finishes successfully.
    """


@dataclass(slots=True, frozen=True)
class ExecutionFailed(Event[Exception]):
    """
    Published when execution terminates due to an exception.
    """


# --------------------------------------------------------------------
# Runtime control
# --------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ExecutionPaused(Event[None]):
    """
    Execution entered the paused state.
    """


@dataclass(slots=True, frozen=True)
class ExecutionResumed(Event[None]):
    """
    Execution resumed.
    """


@dataclass(slots=True, frozen=True)
class ExecutionCancelled(Event[None]):
    """
    Execution cancelled.
    """