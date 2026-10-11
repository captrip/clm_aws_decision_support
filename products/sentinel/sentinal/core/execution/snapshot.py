from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

from .result import ExecutionResult
from .runtime_data import RuntimeData


class ExecutionStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(slots=True, frozen=True)
class ExecutionSnapshot:

    snapshot_id: UUID = field(default_factory=uuid4)

    version: int = 1

    status: ExecutionStatus = ExecutionStatus.CREATED

    attempt: int = 1

    checkpoint: str | None = None

    runtime: RuntimeData = field(default_factory=RuntimeData)

    result: ExecutionResult | None = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )