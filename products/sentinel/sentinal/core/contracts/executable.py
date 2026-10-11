from __future__ import annotations

from abc import ABC, abstractmethod

from sentinal.core.execution import (
    ExecutionContext,
    ExecutionResult,
)


class ExecutableContract(ABC):
    """
    Runtime-neutral executable contract.ß
    """

    @abstractmethod
    async def execute(
        self,
        context: ExecutionContext,
    ) -> ExecutionResult:
        ...