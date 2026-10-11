from __future__ import annotations

from abc import ABC, abstractmethod

from sentinal.core.message import Message


class Actor(ABC):
    """
    Base actor contract.
    """

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def receive(
        self,
        message: Message,
    ) -> None:
        ...