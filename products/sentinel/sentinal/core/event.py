from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .message import Message

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Event(Message[T], Generic[T]):
    """
    Immutable fact.

    Events describe things that have already happened.
    """