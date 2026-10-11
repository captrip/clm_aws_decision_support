from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class ComputeBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    async def dispatch(self, fn: Callable, *args, **kwargs) -> Any: ...

    @abstractmethod
    def profile(self, fn: Callable, *args, **kwargs) -> float: ...
