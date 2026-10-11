from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from sentinal.core.actor_ref import ActorRef
from sentinal.core.message import Message

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Envelope(Generic[T]):
    """
    Wraps a message with delivery metadata.

    Messages remain transport-agnostic.
    The dispatcher uses Envelopes internally.
    """

    message: Message[T]

    recipient: ActorRef