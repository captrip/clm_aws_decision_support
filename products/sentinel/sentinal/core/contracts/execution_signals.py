from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sentinal.core.signal import Signal

@dataclass(slots=True, frozen=True)
class StartExecution(
    Signal[UUID]
):
    """
    Payload contains the execution id.
    """
# @dataclass(slots=True, frozen=True)
# class StartExecution(Signal[None]):
#     """
#     Begin execution.
#     """


# @dataclass(slots=True, frozen=True)
# class PauseExecution(Signal[None]):
#     """
#     Pause execution.
#     """


# @dataclass(slots=True, frozen=True)
# class ResumeExecution(Signal[None]):
#     """
#     Resume execution.
#     """


# @dataclass(slots=True, frozen=True)
# class CancelExecution(Signal[None]):
#     """
#     Cancel execution.
#     """


@dataclass(slots=True, frozen=True)
class RetryExecution(Signal[int]):
    """
    Payload is the retry attempt number.
    """