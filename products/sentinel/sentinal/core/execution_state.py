from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class ExecutionState:

    attempt: int = 1

    status: str = "PENDING"

    runtime_memory: dict[str, Any] = field(
        default_factory=dict
    )

    runtime_instructions: list[str] = field(
        default_factory=list
    )

    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    completed_at: datetime | None = None