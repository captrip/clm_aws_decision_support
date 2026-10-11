from __future__ import annotations

from abc import ABC, abstractmethod

from sentinal.core.observation import Observation
from sentinal.core.signal import Signal


class Governor(ABC):

    @abstractmethod
    async def evaluate(
        self,
        observation: Observation,
    ) -> Signal | None:
        ...