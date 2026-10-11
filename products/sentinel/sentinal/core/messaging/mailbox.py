from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

from .envelop import Envelope

T = TypeVar("T")


class Mailbox(Generic[T]):
    """
    FIFO mailbox owned by a single actor.

    Only the dispatcher writes into the mailbox.
    Only the owning actor consumes from it.
    """

    __slots__ = ("_queue",)

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Envelope[T]] = asyncio.Queue()

    async def put(
        self,
        envelope: Envelope[T],
    ) -> None:
        await self._queue.put(envelope)

    async def get(self) -> Envelope[T]:
        return await self._queue.get()

    def empty(self) -> bool:
        return self._queue.empty()

    def size(self) -> int:
        return self._queue.qsize()