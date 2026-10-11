from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True, frozen=True)
class RuntimeData:
    """
    Immutable runtime-specific information associated with an execution.

    This contains operational metadata that evolves during execution but
    remains immutable within a single snapshot.
    """

    current_node: str | None = None

    retry_count: int = 0

    metrics: Mapping[str, Any] = field(default_factory=dict)

    extras: Mapping[str, Any] = field(default_factory=dict)

    def with_metric(self, key: str, value: Any) -> "RuntimeData":
        metrics = dict(self.metrics)
        metrics[key] = value

        return RuntimeData(
            current_node=self.current_node,
            retry_count=self.retry_count,
            metrics=metrics,
            extras=self.extras,
        )

    def with_extra(self, key: str, value: Any) -> "RuntimeData":
        extras = dict(self.extras)
        extras[key] = value

        return RuntimeData(
            current_node=self.current_node,
            retry_count=self.retry_count,
            metrics=self.metrics,
            extras=extras,
        )