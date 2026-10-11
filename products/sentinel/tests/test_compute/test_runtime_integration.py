import threading
import time

import pytest

from sentinal.compute.device import DevicePreference
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.runtime.runtime import SentinelRuntime


class CPUBoundExecutable(Executable):
    device = DevicePreference.CPU

    async def execute(self, context):
        total = 0.0
        for i in range(2_000_000):
            total += i * 0.000001
        return {"total": total}


class AutoExecutable(Executable):
    async def execute(self, context):
        return {"status": "done", "thread": threading.current_thread().native_id}


class TestRuntimeWithHybridRunner:
    @pytest.mark.asyncio
    async def test_execute_uses_hybrid_runner(self):
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=CPUBoundExecutable(),
            context=ExecutionContext(input={"q": "test"}),
        )

        result = await runtime.execute(session)
        assert result.is_terminal()
        memory = result.state.get_all_memory()
        assert memory.get("actor_status") == "completed"
        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_reports_available_backends(self):
        runtime = SentinelRuntime()
        metrics = runtime.get_metrics()
        assert "backends" in metrics
        assert "cpu" in metrics["backends"]
        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_executor_runs_on_worker_thread(self):
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=AutoExecutable(),
            context=ExecutionContext(input={}),
        )

        result = await runtime.execute(session)
        memory = result.state.get_all_memory()
        worker_tid = memory.get("result", {}).get("thread")
        main_tid = threading.current_thread().native_id

        assert worker_tid is not None
        assert worker_tid != main_tid
        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self):
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=AutoExecutable(),
            context=ExecutionContext(input={}),
        )
        await runtime.execute(session)
        await runtime.shutdown()
        assert runtime._running is False
