from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Generic, TypeVar
from uuid import UUID, uuid4

from .actor_ref import ActorRef

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Message(Generic[T]):
    """
    Immutable transport object exchanged between actors.
    """

    execution_id: UUID

    sender: ActorRef

    data: T

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    message_id: UUID = field(
        default_factory=uuid4
    )