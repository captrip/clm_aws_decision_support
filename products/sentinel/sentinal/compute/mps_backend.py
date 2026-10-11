from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from sentinal.compute.backend import ComputeBackend

logger = logging.getLogger(__name__)

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False


class MPSBackend(ComputeBackend):
    """Dispatches work to Apple GPU via PyTorch MPS."""

    @property
    def name(self) -> str:
        return "mps"

    def is_available(self) -> bool:
        if not _torch_available:
            return False
        return torch.backends.mps.is_available()

    async def dispatch(self, fn: Callable, *args, **kwargs) -> Any:
        logger.debug("Dispatching to MPS GPU")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def profile(self, fn: Callable, *args, **kwargs) -> float:
        if _torch_available:
            torch.mps.synchronize()
        start = time.perf_counter()
        fn(*args, **kwargs)
        if _torch_available:
            torch.mps.synchronize()
        return time.perf_counter() - start
