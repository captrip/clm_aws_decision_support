from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class ExecutionStatus(Enum):
    """
    Represents the lifecycle of an execution.
    """

    CREATED = "created"

    RUNNING = "running"

    PAUSED = "paused"

    CANCELLED = "cancelled"

    FAILED = "failed"

    COMPLETED = "completed"


@dataclass(slots=True, frozen=True)
class ExecutionSnapshot:
    """
    Immutable snapshot of runtime state.

    ExecutionState owns and replaces snapshots over time.
    """

    snapshot_id: UUID = field(default_factory=uuid4)

    version: int = 1

    status: ExecutionStatus = ExecutionStatus.CREATED

    attempt: int = 1

    checkpoint: str | None = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    data: dict[str, Any] = field(default_factory=dict)