from __future__ import annotations

from dataclasses import dataclass

from sentinal.core.observation import Observation


@dataclass(slots=True, frozen=True)
class ExecutionSucceeded(
    Observation[None]
):
    """
    Indicates successful execution.
    """


@dataclass(slots=True, frozen=True)
class ExecutionErrored(
    Observation[Exception]
):
    """
    Indicates execution failure.
    """


@dataclass(slots=True, frozen=True)
class RetrySuggested(
    Observation[int]
):
    """
    Suggest retry.
    """


@dataclass(slots=True, frozen=True)
class RetryLimitReached(
    Observation[int]
):
    """
    Retry exhausted.
    """