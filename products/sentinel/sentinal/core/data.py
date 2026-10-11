from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ------------------------------------------------------------------
# Event Data
# ------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class ExecutionStartedData:
    input: Any


@dataclass(slots=True, frozen=True)
class ExecutionCompletedData:
    output: Any


@dataclass(slots=True, frozen=True)
class ToolStartedData:
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ToolCompletedData:
    tool_name: str
    result: Any
    duration_ms: float


# ------------------------------------------------------------------
# Observation Data
# ------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class LoopDetectedData:
    iterations: int
    reason: str


@dataclass(slots=True, frozen=True)
class ToolFailureData:
    tool_name: str
    error: str


# ------------------------------------------------------------------
# Signal Data
# ------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class RetrySignalData:
    reason: str


@dataclass(slots=True, frozen=True)
class AbortSignalData:
    reason: str


@dataclass(slots=True, frozen=True)
class PauseSignalData:
    reason: str


@dataclass(slots=True, frozen=True)
class InjectInstructionData:
    instruction: str
    priority: int = 100