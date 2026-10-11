from __future__ import annotations

from abc import ABC, abstractmethod

from sentinal.core.event import Event
from sentinal.core.observation import Observation


class Observer(ABC):

    @abstractmethod
    async def observe(
        self,
        event: Event,
    ) -> Observation | None:
        ...