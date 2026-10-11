from __future__ import annotations

from sentinal.core.execution.runtime_data import RuntimeData

from .retry import RetryPolicy


class ExponentialRetryPolicy(RetryPolicy):

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 30.0,
    ) -> None:

        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.multiplier = multiplier
        self.max_delay = max_delay

    def should_retry(
        self,
        runtime: RuntimeData,
        error: Exception,
    ) -> bool:

        return runtime.retry_count < self.max_attempts

    def next_delay(
        self,
        runtime: RuntimeData,
    ) -> float:

        delay = (
            self.base_delay *
            (self.multiplier ** runtime.retry_count)
        )

        return min(delay, self.max_delay)