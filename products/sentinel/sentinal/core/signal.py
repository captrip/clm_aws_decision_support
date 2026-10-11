from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .message import Message

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Signal(Message[T], Generic[T]):
    """
    Future intent produced by a governor.
    """