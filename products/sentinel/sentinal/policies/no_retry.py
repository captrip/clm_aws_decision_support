from __future__ import annotations

from sentinal.core.execution.runtime_data import RuntimeData

from .retry import RetryPolicy


class NoRetryPolicy(RetryPolicy):

    def should_retry(
        self,
        runtime: RuntimeData,
        error: Exception,
    ) -> bool:
        return False

    def next_delay(
        self,
        runtime: RuntimeData,
    ) -> float:
        return 0.0