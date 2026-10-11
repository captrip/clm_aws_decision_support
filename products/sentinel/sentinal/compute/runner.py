from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from sentinal.compute.backend import ComputeBackend
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.compute.profiler import WorkloadProfiler
from sentinal.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

if TYPE_CHECKING:
    from sentinal.actors.executor import Executor
    from sentinal.actors.observer import Observer
    from sentinal.actors.governor import Governor
    from sentinal.core.execution.session import ExecutionSession


class HybridRunner:
    """Routes actor work to CPU threads or GPU based on workload and hardware."""

    def __init__(self, backends: list[ComputeBackend] | None = None):
        if backends is None:
            backends = self.detect_backends()
        self._backends = {b.name: b for b in backends}
        self._profiler = WorkloadProfiler(self._backends)
        self._cpu = self._backends.get("cpu") or CPUBackend()

    @classmethod
    def detect_backends(cls) -> list[ComputeBackend]:
        """Detect available compute backends on this machine."""
        backends: list[ComputeBackend] = [CPUBackend()]

        try:
            from sentinal.compute.mps_backend import MPSBackend
            mps = MPSBackend()
            if mps.is_available():
                backends.append(mps)
        except Exception:
            pass

        try:
            from sentinal.compute.metal_backend import MetalBackend
            metal = MetalBackend()
            if metal.is_available():
                backends.append(metal)
        except Exception:
            pass

        logger.info(f"Detected backends: {[b.name for b in backends]}")
        return backends

    async def run_actors(
        self,
        session: ExecutionSession,
        executor: Executor,
        observer: Observer,
        governor: Governor,
    ) -> ExecutionSession:
        """
        Run all 3 actors with hybrid dispatch.

        - Executor: dispatched to resolved backend (CPU thread or GPU)
        - Observer: CPU thread (polling)
        - Governor: CPU thread (policy checks)
        """
        backend_name = self._profiler.resolve(session.executable)
        backend = self._backends.get(backend_name, self._cpu)

        logger.info(f"Dispatching actors: executor={backend_name}, observer=cpu, governor=cpu")

        async def run_executor():
            await backend.dispatch(
                self._run_sync_actor, executor, session
            )

        async def run_observer():
            await self._cpu.dispatch(
                self._run_observer_sync, observer, session
            )

        async def run_governor():
            await self._cpu.dispatch(
                self._run_governor_sync, governor, session
            )

        await asyncio.gather(
            run_executor(),
            run_observer(),
            run_governor(),
        )

        return session

    def _run_sync_actor(self, executor: Executor, session: ExecutionSession):
        """Run executor's session work synchronously (called from thread)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(executor.execute_session(session))
        finally:
            loop.close()

    def _run_observer_sync(self, observer: Observer, session: ExecutionSession):
        """Run observer polling synchronously (called from thread)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(observer.observe_session(session))
        finally:
            loop.close()

    def _run_governor_sync(self, governor: Governor, session: ExecutionSession):
        """Run governor polling synchronously (called from thread)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(governor.govern_session(session))
        finally:
            loop.close()

    @property
    def available_backends(self) -> list[str]:
        return [name for name, b in self._backends.items() if b.is_available()]

    def shutdown(self):
        for backend in self._backends.values():
            if hasattr(backend, "shutdown"):
                backend.shutdown()
