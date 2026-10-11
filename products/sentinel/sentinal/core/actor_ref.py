from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(slots=True, frozen=True)
class ActorRef:
    """
    Globally unique identifier for an actor.

    An ActorRef represents the identity of an actor rather
    than its implementation.
    """

    name: str

    actor_id: UUID = field(default_factory=uuid4)

    def __str__(self) -> str:
        return f"{self.name}<{self.actor_id}>"