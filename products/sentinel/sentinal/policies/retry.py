from __future__ import annotations

from abc import ABC, abstractmethod

from sentinal.core.execution.runtime_data import RuntimeData


class RetryPolicy(ABC):
    """
    Strategy interface controlling retry behaviour.
    """

    @abstractmethod
    def should_retry(
        self,
        runtime: RuntimeData,
        error: Exception,
    ) -> bool:
        ...

    @abstractmethod
    def next_delay(
        self,
        runtime: RuntimeData,
    ) -> float:
        ...