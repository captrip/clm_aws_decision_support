from __future__ import annotations

from abc import ABC, abstractmethod

from sentinal.core.execution import ExecutionSession


class Runtime(ABC):

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...

    @abstractmethod
    async def submit(
        self,
        session: ExecutionSession,
    ) -> None:
        ...