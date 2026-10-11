"""Judge ownership is per-SESSION, not per-Governor.

The Governor is created once per runtime but `judges=` is per `execute()` call.
These tests pin down that a per-call judge list governs only its own session:
it must not leak into a later run, and two concurrent runs must not clobber
each other.
"""
import asyncio

from sentinal.actors.governor import Governor
from sentinal.core.actor_ref import ActorRef
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.trajectory import TrajectoryStep
from sentinal.judges import judge, redirect
from sentinal.runtime.runtime import SentinelRuntime


class _ShortExecutable(Executable):
    """Emits a handful of steps, ignores corrections, then finishes."""

    def __init__(self, steps: int = 8):
        self._steps = steps

    async def execute(self, context):
        session = context.session
        assert session is not None
        for i in range(self._steps):
            session.append_step(
                TrajectoryStep(index=i, kind="agent", summary=f"s{i}", raw={})
            )
            list(session.drain_corrections())
            await asyncio.sleep(0.02)
        return {"steps": self._steps}


def _spy(name):
    """A judge that records the session ids it was asked about."""
    seen = []

    @judge(name=name)
    def _j(ctx):
        seen.append(ctx.memory.get("session_tag"))
        return None

    _j.seen = seen
    return _j


def _session(tag):
    s = ExecutionSession(
        executable=_ShortExecutable(),
        context=ExecutionContext(input={}),
    )
    s.update_memory("session_tag", tag)
    return s


def _judge_names(governor):
    return [getattr(j, "name", type(j).__name__) for j in governor._judges]


# ── 1. a per-call judges= governs the run even with a pre-registered Governor ──

def test_judges_kwarg_governs_a_pre_registered_governor_run():
    async def go():
        runtime = SentinelRuntime()
        runtime.register(Governor(ActorRef("governor", "local")))
        mine = _spy("mine")
        await runtime.execute(_session("a"), judges=[mine])
        await runtime.shutdown()
        return mine

    mine = asyncio.run(go())
    assert mine.seen, "the judges= passed to execute() should have governed the run"


# ── 2. the sequential leak ────────────────────────────────────────────────────

def test_a_later_run_without_judges_reverts_to_the_default_judge():
    async def go():
        runtime = SentinelRuntime()
        mine = _spy("mine")
        await runtime.execute(_session("first"), judges=[mine])
        governor = next(a for a in runtime.actors if a.ref.name == "governor")
        # Second run passes NO judges= — it must be governed by the default.
        await runtime.execute(_session("second"))
        names = _judge_names(governor)
        await runtime.shutdown()
        return mine, names

    mine, names = asyncio.run(go())
    # The Governor's own judges were never replaced by the per-call override.
    assert names == ["heuristic"], f"judges leaked across runs: {names}"
    # And the first run's judge was not consulted about the second session.
    assert set(mine.seen) <= {"first"}, f"judge leaked into run 2: {mine.seen}"


# ── 3. the concurrent race ────────────────────────────────────────────────────

def test_two_concurrent_runs_are_each_governed_only_by_their_own_judge():
    async def go():
        runtime = SentinelRuntime()
        a = _spy("judge_a")
        b = _spy("judge_b")
        sa, sb = _session("A"), _session("B")
        await asyncio.gather(
            runtime.execute(sa, judges=[a]),
            runtime.execute(sb, judges=[b]),
        )
        await runtime.shutdown()
        return a, b

    a, b = asyncio.run(go())
    assert a.seen, "judge_a was never consulted"
    assert b.seen, "judge_b was never consulted"
    assert set(a.seen) == {"A"}, f"judge_a saw foreign sessions: {set(a.seen)}"
    assert set(b.seen) == {"B"}, f"judge_b saw foreign sessions: {set(b.seen)}"


# ── set_judges() stays supported ──────────────────────────────────────────────

def test_set_judges_still_replaces_the_governors_own_judges():
    @judge(name="explicit")
    def explicit(ctx):
        return redirect("nudge")

    gov = Governor(ActorRef("governor", "local"))
    gov.set_judges([explicit])
    assert _judge_names(gov) == ["explicit"]


def test_a_session_override_beats_the_governors_own_judges():
    @judge(name="default_judge")
    def default_judge(ctx):
        return None

    @judge(name="override")
    def override(ctx):
        return None

    gov = Governor(ActorRef("governor", "local"), judges=[default_judge])
    gov.set_session_judges("sess-1", [override])
    assert [j.name for j in gov._resolve_judges("sess-1")] == ["override"]
    assert [j.name for j in gov._resolve_judges("sess-2")] == ["default_judge"]


def test_the_session_override_is_cleaned_up_after_governing():
    @judge(name="override")
    def override(ctx):
        return None

    class _Session:
        def __init__(self, polls):
            self.session_id = "sess-cleanup"
            self._polls = polls
            self.state = self

        def get_trajectory(self):
            return []

        def get_all_memory(self):
            return {}

        def push_correction(self, c):
            pass

        def is_terminal(self):
            self._polls -= 1
            return self._polls <= 0

    gov = Governor(ActorRef("governor", "local"))
    gov.set_session_judges("sess-cleanup", [override])
    asyncio.run(gov.govern_session(_Session(polls=3), poll_interval=0.001))
    assert gov._session_judges == {}, "per-session judges must not accumulate"
