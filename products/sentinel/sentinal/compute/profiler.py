from __future__ import annotations

import inspect
import logging
from typing import Any

from sentinal.compute.backend import ComputeBackend
from sentinal.compute.device import DevicePreference

logger = logging.getLogger(__name__)


class WorkloadProfiler:
    """Determines the best compute backend for an executable via heuristics and profiling."""

    def __init__(self, backends: dict[str, ComputeBackend]):
        self._backends = backends
        self._cache: dict[type, str] = {}

    def resolve(self, executable) -> str:
        """Determine the best backend name for this executable."""
        exe_type = type(executable)
        if exe_type in self._cache:
            return self._cache[exe_type]

        # Check explicit device preference
        device = getattr(executable, "device", None)
        if isinstance(device, DevicePreference):
            if device == DevicePreference.CPU:
                self._cache[exe_type] = "cpu"
                logger.debug(f"Resolved {exe_type.__name__} -> cpu")
                return "cpu"
            elif device == DevicePreference.GPU:
                result = self._resolve_gpu()
                self._cache[exe_type] = result
                logger.debug(f"Resolved {exe_type.__name__} -> {result}")
                return result

        # AUTO or no device set: use heuristic
        if self.uses_tensors(executable):
            result = self._resolve_gpu()
        else:
            result = "cpu"

        self._cache[exe_type] = result
        logger.debug(f"Resolved {exe_type.__name__} -> {result}")
        return result

    def uses_tensors(self, executable) -> bool:
        """Check if executable likely uses torch tensors."""
        try:
            source = inspect.getsource(type(executable))
            tensor_indicators = ["torch", "tensor", ".mps", ".cuda", "device="]
            return any(ind in source for ind in tensor_indicators)
        except (OSError, TypeError):
            return False

    def profile_and_cache(self, executable, test_fn) -> str:
        """Run a microbenchmark on available backends and cache the winner."""
        best_name = "cpu"
        best_time = float("inf")

        for name, backend in self._backends.items():
            if not backend.is_available():
                continue
            try:
                elapsed = backend.profile(test_fn)
                if elapsed < best_time:
                    best_time = elapsed
                    best_name = name
            except Exception:
                continue

        self._cache[type(executable)] = best_name
        return best_name

    def _resolve_gpu(self) -> str:
        """Find the best available GPU backend, fall back to cpu."""
        for name in ("mps", "metal"):
            if name in self._backends and self._backends[name].is_available():
                return name
        return "cpu"
