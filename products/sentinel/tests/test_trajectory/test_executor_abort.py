import asyncio
from sentinal.core.actor_ref import ActorRef
from sentinal.actors.executor import Executor
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.execution.snapshot import ExecutionStatus
from sentinal.exceptions.runtime import ExecutionAborted


class _AbortingExecutable(Executable):
    async def execute(self, context):
        # prove the live session was attached to the context
        assert context.session is not None
        raise ExecutionAborted("stopped by governor")


class _OkExecutable(Executable):
    async def execute(self, context):
        assert context.session is not None
        return {"ok": True}


async def _noop(*args, **kwargs):
    """No-op publish for testing without a dispatcher."""
    pass


def test_executor_marks_aborted_on_execution_aborted():
    session = ExecutionSession(executable=_AbortingExecutable(),
                               context=ExecutionContext(input={}))
    ex = Executor(ActorRef("executor", "local"))
    ex.publish = _noop  # Mock publish to avoid dispatcher requirement
    asyncio.run(ex.execute_session(session))
    assert session.state.status is ExecutionStatus.ABORTED
    assert session.get_memory("actor_status") == "aborted"


def test_executor_attaches_session_to_context_on_success():
    session = ExecutionSession(executable=_OkExecutable(),
                               context=ExecutionContext(input={}))
    ex = Executor(ActorRef("executor", "local"))
    ex.publish = _noop  # Mock publish to avoid dispatcher requirement
    asyncio.run(ex.execute_session(session))
    assert session.state.status is ExecutionStatus.COMPLETED
