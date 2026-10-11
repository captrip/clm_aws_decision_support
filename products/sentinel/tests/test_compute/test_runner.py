import asyncio
import threading
import time

import pytest

from sentinal.compute.runner import HybridRunner
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.compute.device import DevicePreference
from sentinal.core.actor_ref import ActorRef
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.actors.executor import Executor
from sentinal.actors.observer import Observer
from sentinal.actors.governor import Governor


class NoOpDispatcher:
    """Minimal dispatcher that does nothing - for testing without full runtime."""
    async def publish(self, message):
        pass

    def register(self, ref, mailbox):
        pass


class SlowCPUExecutable(Executable):
    device = DevicePreference.CPU

    async def execute(self, context):
        total = 0.0
        for i in range(2_000_000):
            total += i * 0.000001
        return {"total": total}


class TestHybridRunner:
    @pytest.fixture
    def runner(self):
        r = HybridRunner(backends=[CPUBackend(max_workers=4)])
        yield r
        r.shutdown()

    def test_detect_backends(self):
        backends = HybridRunner.detect_backends()
        names = [b.name for b in backends]
        assert "cpu" in names

    @pytest.mark.asyncio
    async def test_run_actors_completes(self, runner):
        session = ExecutionSession(
            executable=SlowCPUExecutable(),
            context=ExecutionContext(input={"query": "test"}),
        )

        executor = Executor(ActorRef("executor", "local"))
        observer = Observer(ActorRef("observer", "local"))
        governor = Governor(ActorRef("governor", "local"))

        # Set up no-op dispatcher for all actors
        noop = NoOpDispatcher()
        executor.dispatcher = noop
        observer.dispatcher = noop
        governor.dispatcher = noop

        result = await runner.run_actors(session, executor, observer, governor)
        assert result.is_terminal()

    @pytest.mark.asyncio
    async def test_run_actors_produces_result(self, runner):
        session = ExecutionSession(
            executable=SlowCPUExecutable(),
            context=ExecutionContext(input={}),
        )

        executor = Executor(ActorRef("executor", "local"))
        observer = Observer(ActorRef("observer", "local"))
        governor = Governor(ActorRef("governor", "local"))

        # Set up no-op dispatcher for all actors
        noop = NoOpDispatcher()
        executor.dispatcher = noop
        observer.dispatcher = noop
        governor.dispatcher = noop

        result = await runner.run_actors(session, executor, observer, governor)
        memory = result.state.get_all_memory()
        assert memory.get("actor_status") == "completed"
        assert "result" in memory
