from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from sentinal.core.actor_ref import ActorRef
from sentinal.core.message import Message
from sentinal.core.messaging.dispatcher import DefaultDispatcher
from sentinal.core.messaging.mailbox import Mailbox


class BaseActor(ABC):
    """
    Base class for every Sentinel actor.
    """

    def __init__(
        self,
        ref: ActorRef,
    ) -> None:

        self.ref = ref

        self.mailbox = Mailbox()

        self.dispatcher: DefaultDispatcher | None = None

        self._running = False

        self._task: asyncio.Task | None = None

    async def start(self) -> None:

        if self._running:
            return

        self._running = True

        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:

        self._running = False

        if self._task is not None:

            self._task.cancel()

            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def publish(
        self,
        message: Message,
    ) -> None:

        if self.dispatcher is None:
            raise RuntimeError(
                "Actor is not attached to a dispatcher."
            )

        await self.dispatcher.publish(message)

    async def _run(self) -> None:

        while self._running:

            envelope = await self.mailbox.get()

            await self.receive(
                envelope.message
            )

    @abstractmethod
    async def receive(
        self,
        message: Message,
    ) -> None:
        """
        Handle an incoming message.
        """