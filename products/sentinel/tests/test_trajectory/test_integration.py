# tests/test_trajectory/test_integration.py
import asyncio
from sentinal.runtime.runtime import SentinelRuntime
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.execution.snapshot import ExecutionStatus
from sentinal.core.trajectory import TrajectoryStep
from sentinal.exceptions.runtime import ExecutionAborted
from sentinal.actors.judges import HeuristicJudge


class _StreamingExecutable(Executable):
    """Emits many identical tool steps; drains corrections; aborts on STOP,
    finalizes on FORCE_FINALIZE."""
    async def execute(self, context):
        session = context.session
        assert session is not None
        i = 0
        while i < 50:
            session.append_step(
                TrajectoryStep(index=i, kind="tool", summary="web_search", raw={})
            )
            for c in session.drain_corrections():
                if c.flag.value == "stop":
                    raise ExecutionAborted(c.note or "stop")
                if c.flag.value in ("force_finalize", "avoid_tool", "redirect"):
                    return {"finalized_at": i, "flag": c.flag.value}
            await asyncio.sleep(0.01)
            i += 1
        return {"finalized_at": i}


async def _run():
    runtime = SentinelRuntime()
    session = ExecutionSession(
        executable=_StreamingExecutable(),
        context=ExecutionContext(input={}),
    )
    await runtime.execute(session)
    await runtime.shutdown()
    return session


def test_correction_flows_end_to_end_and_finalizes_early():
    session = asyncio.run(_run())
    memory = session.state.get_all_memory()
    result = memory.get("result", {})
    # HeuristicJudge fires AVOID_TOOL after 3 identical tool calls, so the
    # executable returns well before its 50-step cap.
    # The Governor polls at 0.1s intervals, so allow time for it to observe
    # the trajectory and push the correction (typically arrives by ~step 20).
    assert result.get("finalized_at", 999) < 25
    assert result.get("flag") in ("avoid_tool", "redirect", "force_finalize")


class _LongIgnoringExecutable(Executable):
    """Emits ~30 identical tool steps but IGNORES corrections, driving
    trajectory length (and Governor's _last_judged_len) high."""
    async def execute(self, context):
        session = context.session
        assert session is not None
        for i in range(30):
            session.append_step(
                TrajectoryStep(index=i, kind="tool", summary="web_search", raw={})
            )
            # Drain corrections but don't act on them — keep running
            list(session.drain_corrections())
            await asyncio.sleep(0.01)
        return {"finalized_at": 30, "flag": "completed_full_loop"}


async def _run_two_sessions():
    runtime = SentinelRuntime()
    # Session 1: long trajectory (~30 steps), ignores corrections
    session1 = ExecutionSession(
        executable=_LongIgnoringExecutable(),
        context=ExecutionContext(input={}),
    )
    await runtime.execute(session1)

    # Session 2: normal executable that returns early on correction
    session2 = ExecutionSession(
        executable=_StreamingExecutable(),
        context=ExecutionContext(input={}),
    )
    await runtime.execute(session2)
    await runtime.shutdown()
    return session1, session2


class _RestartingExecutable(Executable):
    """Mirrors the search_agent example's corrected control flow: an outer loop
    that restarts a bounded inner "stream" on REDIRECT/AVOID_TOOL, stops on
    FORCE_FINALIZE, and aborts on STOP. Steps carry DISTINCT summaries so the
    HeuristicJudge's repeat/loop rules never fire — only rule (1), the step
    budget (n > 20), can produce a FORCE_FINALIZE. This guards the regression
    where FORCE_FINALIZE was a no-op and the trajectory ran to its full cap."""
    async def execute(self, context):
        session = context.session
        assert session is not None
        i = 0
        finalized = False
        while True:
            restart = False
            for _ in range(200):  # bounded inner "stream"
                session.append_step(
                    TrajectoryStep(index=i, kind="agent", summary=f"step-{i}", raw={})
                )
                for c in session.drain_corrections():
                    if c.flag.value == "stop":
                        raise ExecutionAborted(c.note or "stop")
                    if c.flag.value == "force_finalize":
                        finalized = True
                    elif c.flag.value in ("redirect", "avoid_tool"):
                        restart = True
                if finalized or restart:
                    break
                await asyncio.sleep(0.01)
                i += 1
            if finalized or not restart:
                break
        return {"finalized_at": i, "flag": "force_finalize" if finalized else "completed"}


async def _run_executable(executable):
    runtime = SentinelRuntime()
    session = ExecutionSession(
        executable=executable,
        context=ExecutionContext(input={}),
    )
    await runtime.execute(session)
    await runtime.shutdown()
    return session


def test_force_finalize_halts_trajectory_near_step_budget():
    session = asyncio.run(_run_executable(_RestartingExecutable()))
    result = session.state.get_all_memory().get("result", {})
    # With FORCE_FINALIZE honored, the run stops once the trajectory passes the
    # judge's step_budget (20) — not at the 200-step inner cap. The Governor
    # polls every 0.1s while steps arrive every 0.01s, so finalization lands a
    # bit past 20; the key guarantee is it's bounded far below the cap.
    assert result.get("flag") == "force_finalize"
    assert 20 < result.get("finalized_at", 999) < 60


def test_steering_works_on_second_session_of_reused_runtime():
    session1, session2 = asyncio.run(_run_two_sessions())

    # Session 1 leaves _last_judged_len at ~30. Without the line-96 reset,
    # session 2's Governor starts its cursor at ~30, so it doesn't judge
    # session 2 until after step 30, meaning AVOID_TOOL (3 identical tool
    # calls) fires ~step 33, giving finalized_at >= 30, breaking this < 25
    # assertion. WITH the reset, session 2 is judged from step 0 and
    # finalizes early (~step 3-6), passing the assertion.
    memory2 = session2.state.get_all_memory()
    result2 = memory2.get("result", {})

    assert result2.get("finalized_at", 999) < 25
    assert result2.get("flag") in ("avoid_tool", "redirect", "force_finalize")
