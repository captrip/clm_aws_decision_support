from __future__ import annotations

from abc import ABC, abstractmethod

from sentinal.core.message import Message


class Dispatcher(ABC):

    @abstractmethod
    async def publish(
        self,
        message: Message,
    ) -> None:
        ...