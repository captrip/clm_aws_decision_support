from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .message import Message

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Observation(Message[T], Generic[T]):
    """
    Produced by observers after analysing events.
    """