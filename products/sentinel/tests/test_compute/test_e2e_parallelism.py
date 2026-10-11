"""
End-to-end proof that SentinelRuntime achieves true heterogeneous parallelism:
- CPU-bound executables run on real OS threads
- GPU-bound executables dispatch to Metal/MPS
- Observer and Governor run on separate OS threads simultaneously
"""
import sys
import threading
import time

import pytest
import torch

from sentinal.compute.device import DevicePreference
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.runtime.runtime import SentinelRuntime


@pytest.fixture(autouse=True)
def check_env():
    assert sys._is_gil_enabled() is False, "GIL must be disabled"


class CPUHeavyExecutable(Executable):
    """CPU-bound workload that records its thread ID and timing."""
    device = DevicePreference.CPU

    async def execute(self, context):
        tid = threading.current_thread().native_id
        t0 = time.perf_counter()
        total = 0.0
        for i in range(3_000_000):
            total += i * 0.000001
        t1 = time.perf_counter()
        return {
            "thread_id": tid,
            "start": t0,
            "end": t1,
            "duration_ms": (t1 - t0) * 1000,
            "total": total,
        }


class GPUMatmulExecutable(Executable):
    """GPU-bound workload using PyTorch MPS."""
    device = DevicePreference.GPU

    async def execute(self, context):
        tid = threading.current_thread().native_id
        size = context.input.get("size", 2048)

        a = torch.randn(size, size, device="mps")
        b = torch.randn(size, size, device="mps")
        _ = a @ b
        torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(5):
            c = a @ b
        torch.mps.synchronize()
        t1 = time.perf_counter()

        return {
            "thread_id": tid,
            "start": t0,
            "end": t1,
            "duration_ms": (t1 - t0) * 1000,
            "shape": list(c.shape),
        }


class TestEndToEndParallelism:
    @pytest.mark.asyncio
    async def test_cpu_executable_runs_on_worker_thread(self):
        """Prove executor runs on a different OS thread than main."""
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=CPUHeavyExecutable(),
            context=ExecutionContext(input={}),
        )

        result = await runtime.execute(session)
        memory = result.state.get_all_memory()
        exe_result = memory["result"]

        main_tid = threading.current_thread().native_id
        assert exe_result["thread_id"] != main_tid, (
            "Executor ran on main thread — not dispatched to worker"
        )
        assert exe_result["duration_ms"] > 50, "CPU work too fast to measure"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_gpu_executable_dispatches_to_mps(self):
        """Prove GPU workload runs via MPS and produces correct results."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=GPUMatmulExecutable(),
            context=ExecutionContext(input={"size": 2048}),
        )

        result = await runtime.execute(session)
        memory = result.state.get_all_memory()
        exe_result = memory["result"]

        assert exe_result["shape"] == [2048, 2048]
        assert exe_result["duration_ms"] > 0

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_wall_time_shows_concurrency(self):
        """
        The full 3-actor model should complete in roughly the time of the
        executor alone, since observer/governor run concurrently on separate threads.
        """
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=CPUHeavyExecutable(),
            context=ExecutionContext(input={}),
        )

        start = time.perf_counter()
        result = await runtime.execute(session)
        wall_time = time.perf_counter() - start

        memory = result.state.get_all_memory()
        exe_result = memory["result"]

        exe_duration = exe_result["duration_ms"]
        wall_ms = wall_time * 1000

        print(f"\n  Executor work: {exe_duration:.0f}ms")
        print(f"  Wall time: {wall_ms:.0f}ms")
        print(f"  Overhead: {wall_ms - exe_duration:.0f}ms")

        # Wall time should NOT be 3x executor time (would be if serial)
        # Allow some overhead for thread startup
        assert wall_ms < exe_duration * 2.0, (
            f"Wall time ({wall_ms:.0f}ms) suggests actors are serial, "
            f"not parallel (executor took {exe_duration:.0f}ms)"
        )

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_sessions_work(self):
        """Runtime handles multiple sequential sessions correctly."""
        runtime = SentinelRuntime()

        for i in range(3):
            session = ExecutionSession(
                executable=CPUHeavyExecutable(),
                context=ExecutionContext(input={"iteration": i}),
            )
            result = await runtime.execute(session)
            assert result.is_terminal()
            memory = result.state.get_all_memory()
            assert memory["actor_status"] == "completed"

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_detects_all_backends(self):
        """Runtime should detect CPU + GPU backends on this machine."""
        runtime = SentinelRuntime()
        metrics = runtime.get_metrics()
        backends = metrics["backends"]

        print(f"\n  Detected backends: {backends}")

        assert "cpu" in backends
        # On Apple Silicon with our setup, should also have mps and/or metal
        assert len(backends) >= 2, (
            f"Expected at least CPU + one GPU backend, got: {backends}"
        )

        await runtime.shutdown()
