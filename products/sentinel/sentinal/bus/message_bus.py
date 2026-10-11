from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sentinal.core.message import Message

T = TypeVar("T", bound=Message)

Handler = Callable[[T], Awaitable[None]]


class MessageBus:

    def __init__(self) -> None:

        self._subscriptions: dict[type, list[Handler]] = defaultdict(list)

        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        message_type: type[T],
        handler: Handler[T],
    ) -> None:

        async with self._lock:
            self._subscriptions[message_type].append(handler)

    async def publish(self, message: Message) -> None:

        handlers = []

        async with self._lock:

            for cls, subscribers in self._subscriptions.items():

                if isinstance(message, cls):

                    handlers.extend(subscribers)

        if not handlers:
            return

        async with asyncio.TaskGroup() as tg:

            for handler in handlers:
                tg.create_task(handler(message))