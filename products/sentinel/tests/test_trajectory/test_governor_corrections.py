# tests/test_trajectory/test_governor_corrections.py
import asyncio
from sentinal.core.actor_ref import ActorRef
from sentinal.actors.governor import Governor
from sentinal.core.trajectory import (
    Correction, CorrectionFlag, TrajectoryStep,
)
from sentinal.telemetry.dashboard import activity_bus


class _FakeJudge:
    def __init__(self, correction):
        self._c = correction
        self.calls = 0

    def judge(self, trajectory):
        self.calls += 1
        return self._c


class _Session:
    """Fake session exposing the channels the Governor uses."""
    def __init__(self, steps, polls):
        self.session_id = "sess-gov"
        self._steps = steps
        self._pushed = []
        self._polls = polls
        self.state = self

    def get_trajectory(self):
        return list(self._steps)

    def push_correction(self, c):
        self._pushed.append(c)

    def get_all_memory(self):
        return {}

    def is_terminal(self):
        self._polls -= 1
        return self._polls <= 0


def _run_governor(session, judge, poll_interval=0.001, **kw):
    gov = Governor(ActorRef("governor", "local"), judge=judge, **kw)
    asyncio.run(gov.govern_session(session, poll_interval=poll_interval))
    return gov


def test_governor_pushes_non_continue_correction():
    steps = [TrajectoryStep(index=0, kind="tool", summary="web_search", raw={})]
    judge = _FakeJudge(Correction(flag=CorrectionFlag.REDIRECT, note="try X"))
    session = _Session(steps, polls=3)
    _run_governor(session, judge)
    assert any(c.flag is CorrectionFlag.REDIRECT for c in session._pushed)


def test_governor_does_not_push_continue():
    steps = [TrajectoryStep(index=0, kind="agent", summary="ok", raw={})]
    judge = _FakeJudge(Correction(flag=CorrectionFlag.CONTINUE))
    session = _Session(steps, polls=3)
    _run_governor(session, judge)
    assert session._pushed == []


def test_governor_does_not_re_push_on_unchanged_trajectory():
    steps = [TrajectoryStep(index=0, kind="tool", summary="web_search", raw={})]
    judge = _FakeJudge(Correction(flag=CorrectionFlag.REDIRECT))
    # The session must OUTLIVE the Governor's rejudge_interval (default 1.0s),
    # or the time-based re-judgment trigger never fires and the test asserts
    # nothing: 30 polls x 0.05s ~= 1.5s.
    session = _Session(steps, polls=30)
    gov = _run_governor(session, judge, poll_interval=0.05)
    # The stateless judge is re-asked (the stale trigger fires past 1.0s) and
    # returns the same verdict, but an identical correction is never re-pushed.
    assert judge.calls > 1, "the stale trigger should have re-asked the judge"
    assert len(session._pushed) == 1
    # The suppressed repeat is not recorded either.
    corrections = [i for i in gov.get_interventions() if i["type"] == "correction"]
    assert len(corrections) == 1


def test_governor_survives_judge_exception():
    class _Boom:
        def judge(self, trajectory):
            raise RuntimeError("judge broke")
    steps = [TrajectoryStep(index=0, kind="tool", summary="x", raw={})]
    session = _Session(steps, polls=3)
    _run_governor(session, _Boom())   # must not raise
    assert session._pushed == []
