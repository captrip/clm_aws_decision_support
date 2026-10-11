from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class ExecutionResult(Generic[T]):
    """
    Final result returned by an Executable.
    """

    output: T

    metadata: dict[str, object] | None = None