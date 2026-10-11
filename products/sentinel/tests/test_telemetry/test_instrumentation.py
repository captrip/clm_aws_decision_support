import logging

import pytest

from sentinal.compute.device import DevicePreference
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.runtime.runtime import SentinelRuntime
from sentinal.telemetry import configure


class SimpleExecutable(Executable):
    device = DevicePreference.CPU

    async def execute(self, context):
        return {"status": "done"}


class TestInstrumentation:
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        configure(log_level="DEBUG", trace_exporter="none")
        yield
        logger = logging.getLogger("sentinal")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    @pytest.mark.asyncio
    async def test_runtime_logs_execution(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="sentinal"):
            runtime = SentinelRuntime()
            session = ExecutionSession(
                executable=SimpleExecutable(),
                context=ExecutionContext(input={}),
            )
            await runtime.execute(session)
            await runtime.shutdown()

        messages = " ".join(r.message for r in caplog.records)
        assert "session" in messages.lower() or "executing" in messages.lower()

    @pytest.mark.asyncio
    async def test_runner_logs_dispatch(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="sentinal"):
            runtime = SentinelRuntime()
            session = ExecutionSession(
                executable=SimpleExecutable(),
                context=ExecutionContext(input={}),
            )
            await runtime.execute(session)
            await runtime.shutdown()

        loggers_used = set(r.name for r in caplog.records)
        assert any("runtime" in name or "runner" in name for name in loggers_used)

    @pytest.mark.asyncio
    async def test_no_print_statements_in_output(self, capsys):
        """After instrumentation, actors should use logging not print."""
        configure(log_level="WARNING", trace_exporter="none")
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=SimpleExecutable(),
            context=ExecutionContext(input={}),
        )
        await runtime.execute(session)
        await runtime.shutdown()

        captured = capsys.readouterr()
        # No emoji print statements should appear in stdout
        assert "👁️" not in captured.out
        assert "👑" not in captured.out
