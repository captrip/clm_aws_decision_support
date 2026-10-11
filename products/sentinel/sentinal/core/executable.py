from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.result import ExecutionResult

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class Executable(ABC, Generic[TInput, TOutput]):
    """
    Base interface implemented by all executable workloads.

    Set `device` class attribute to control compute routing:
    - DevicePreference.CPU: force OS thread
    - DevicePreference.GPU: prefer GPU (MPS/Metal)
    - DevicePreference.AUTO or None: runtime decides (default)

    Examples:

    - LangGraph adapter
    - CrewAI adapter
    - AutoGen adapter
    - Custom Python workflow
    """

    device = None  # Subclasses can set to DevicePreference.CPU/GPU/AUTO

    @abstractmethod
    async def execute(
        self,
        context: ExecutionContext,
    ) -> ExecutionResult[TOutput]:
        """
        Execute the workload.

        Implementations should remain stateless.
        """
        raise NotImplementedError