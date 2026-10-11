from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from sentinal.compute.backend import ComputeBackend

logger = logging.getLogger(__name__)


class CPUBackend(ComputeBackend):
    """Dispatches work to real OS threads via ThreadPoolExecutor."""

    def __init__(self, max_workers: int | None = None):
        workers = max_workers or max(1, (os.cpu_count() or 4) - 1)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._shutdown = False

    @property
    def name(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        return True

    async def dispatch(self, fn: Callable, *args, **kwargs) -> Any:
        if self._shutdown:
            raise RuntimeError("CPUBackend has been shut down")
        logger.debug("Dispatching to CPU thread pool")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, lambda: fn(*args, **kwargs)
        )

    def profile(self, fn: Callable, *args, **kwargs) -> float:
        start = time.perf_counter()
        fn(*args, **kwargs)
        return time.perf_counter() - start

    def shutdown(self) -> None:
        self._shutdown = True
        self._pool.shutdown(wait=False)
