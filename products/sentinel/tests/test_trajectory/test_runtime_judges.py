import asyncio

import pytest

from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.trajectory import TrajectoryStep
from sentinal.exceptions.runtime import ExecutionAborted
from sentinal.judges import judge, stop
from sentinal.runtime.runtime import SentinelRuntime


class _CountingExecutable(Executable):
    """Emits steps until stopped; records where it stopped."""
    async def execute(self, context):
        session = context.session
        assert session is not None
        i = 0
        while i < 50:
            session.append_step(
                TrajectoryStep(index=i, kind="agent", summary=f"step-{i}", raw={})
            )
            for c in session.drain_corrections():
                if c.flag.value == "stop":
                    raise ExecutionAborted(c.note or "stop")
            await asyncio.sleep(0.01)
            i += 1
        return {"finalized_at": i}


async def _run_with_judges(judges):
    runtime = SentinelRuntime()
    session = ExecutionSession(
        executable=_CountingExecutable(),
        context=ExecutionContext(input={"query": "hello"}),
    )
    await runtime.execute(session, judges=judges)
    governor = next(a for a in runtime.actors if a.ref.name == "governor")
    interventions = governor.get_interventions()
    await runtime.shutdown()
    return session, interventions


def test_custom_business_judge_attached_via_judges_kwarg_steers_the_run():
    @judge(name="business_metric")
    def business_metric(ctx):
        # A stand-in business rule: abort once the run passes 3 steps.
        if len(ctx) > 3:
            return stop("business rule tripped", meta={"steps": len(ctx)})
        return None

    session, interventions = asyncio.run(_run_with_judges([business_metric]))

    assert session.state.status.value == "aborted"
    corrections = [i for i in interventions if i["type"] == "correction"]
    assert corrections, "the custom judge should have been consulted"
    assert corrections[0]["judge"] == "business_metric"
    assert corrections[0]["flag"] == "stop"
    assert corrections[0]["meta"]["steps"] > 3


def test_judges_see_the_execution_input():
    seen = {}

    @judge(name="input_probe")
    def input_probe(ctx):
        seen.update(ctx.input)
        return stop("done probing")

    asyncio.run(_run_with_judges([input_probe]))
    assert seen.get("query") == "hello"
