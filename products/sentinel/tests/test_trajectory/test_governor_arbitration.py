import asyncio

import pytest

from sentinal.core.actor_ref import ActorRef
from sentinal.actors.governor import Governor
from sentinal.core.trajectory import Correction, CorrectionFlag, TrajectoryStep
from sentinal.judges import (
    LatencyBudgetJudge, RepeatToolJudge, StepBudgetJudge, finalize, judge,
    redirect, stop,
)


class _Session:
    """Fake session exposing the channels the Governor uses."""
    def __init__(self, steps, polls):
        self.session_id = "sess-arb"
        self._steps = list(steps)
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


def _steps(n=1):
    return [TrajectoryStep(index=i, kind="tool", summary="web_search", raw={})
            for i in range(n)]


def _run(session, **kw):
    gov = Governor(ActorRef("governor", "local"), **kw)
    asyncio.run(gov.govern_session(session, poll_interval=0.001))
    return gov


def _corrections(gov):
    return [i for i in gov.get_interventions() if i["type"] == "correction"]


def test_most_severe_correction_wins():
    @judge(name="mild")
    def mild(ctx):
        return redirect("nudge")

    @judge(name="severe")
    def severe(ctx):
        return stop("halt")

    session = _Session(_steps(), polls=3)
    _run(session, judges=[mild, severe])
    assert [c.flag for c in session._pushed] == [CorrectionFlag.STOP]


def test_severity_wins_regardless_of_registration_order():
    @judge(name="mild")
    def mild(ctx):
        return redirect("nudge")

    @judge(name="severe")
    def severe(ctx):
        return stop("halt")

    session = _Session(_steps(), polls=3)
    _run(session, judges=[severe, mild])   # severe first this time
    assert [c.flag for c in session._pushed] == [CorrectionFlag.STOP]


def test_all_findings_are_recorded_and_only_the_winner_is_pushed():
    @judge(name="mild")
    def mild(ctx):
        return redirect("nudge")

    @judge(name="mid")
    def mid(ctx):
        return finalize("wrap up")

    @judge(name="severe")
    def severe(ctx):
        return stop("halt")

    session = _Session(_steps(), polls=3)
    gov = _run(session, judges=[mild, mid, severe])
    recorded = _corrections(gov)
    assert {c["judge"] for c in recorded} == {"mild", "mid", "severe"}
    pushed = [c for c in recorded if c["pushed"]]
    assert len(pushed) == 1
    assert pushed[0]["judge"] == "severe"


def test_ties_break_by_registration_order():
    @judge(name="first")
    def first(ctx):
        return redirect("a")

    @judge(name="second")
    def second(ctx):
        return redirect("b")

    session = _Session(_steps(), polls=3)
    _run(session, judges=[first, second])
    assert session._pushed[0].note == "a"


def test_a_raising_judge_does_not_suppress_its_peers():
    class _Boom:
        name = "boom"

        def judge(self, ctx):
            raise RuntimeError("judge broke")

    @judge(name="good")
    def good(ctx):
        return redirect("still works")

    session = _Session(_steps(), polls=3)
    gov = _run(session, judges=[_Boom(), good])
    assert [c.flag for c in session._pushed] == [CorrectionFlag.REDIRECT]
    assert [c["judge"] for c in _corrections(gov)] == ["good"]


def test_a_vandal_judge_cannot_blind_its_peers():
    # Fault isolation is per-judge: a judge that tries to mutate the trajectory
    # must not change what its peers see.
    @judge(name="vandal")
    def vandal(ctx):
        try:
            ctx.trajectory.clear()
        except (AttributeError, TypeError):
            pass
        return None

    session = _Session(_steps(25), polls=3)
    _run(session, judges=[vandal, StepBudgetJudge(max_steps=20)])
    assert [c.flag for c in session._pushed] == [CorrectionFlag.FORCE_FINALIZE]


def test_a_judge_returning_a_non_correction_is_a_no_op():
    @judge(name="bogus")
    def bogus(ctx):
        return "not a correction"

    session = _Session(_steps(), polls=3)
    _run(session, judges=[bogus])   # must not raise
    assert session._pushed == []


def test_a_judge_returning_a_str_flag_correction_is_a_no_op():
    # CorrectionFlag is a str Enum, so Correction(flag="stop") passes an
    # isinstance(c, Correction) guard and used to crash arbitration with
    # AttributeError: 'str' object has no attribute 'value'.
    @judge(name="str_flag")
    def str_flag(ctx):
        return Correction(flag="stop")

    session = _Session(_steps(), polls=3)
    gov = _run(session, judges=[str_flag])   # must not raise
    assert session._pushed == []
    assert _corrections(gov) == []


def test_a_judge_returning_an_unknown_flag_is_a_no_op():
    # An unknown flag string is not in SEVERITY and used to crash arbitration
    # with KeyError: 'abort_now'.
    @judge(name="unknown_flag")
    def unknown_flag(ctx):
        return Correction(flag="abort_now")

    session = _Session(_steps(), polls=3)
    gov = _run(session, judges=[unknown_flag])   # must not raise
    assert session._pushed == []
    assert _corrections(gov) == []


def test_a_judge_with_a_bad_flag_does_not_suppress_its_peers():
    @judge(name="bad")
    def bad(ctx):
        return Correction(flag="abort_now")

    @judge(name="good")
    def good(ctx):
        return redirect("still works")

    session = _Session(_steps(), polls=3)
    gov = _run(session, judges=[bad, good])
    assert [c.flag for c in session._pushed] == [CorrectionFlag.REDIRECT]
    assert [c["judge"] for c in _corrections(gov)] == ["good"]


def test_meta_is_carried_into_the_interventions_log():
    @judge(name="cost_cap")
    def cost_cap(ctx):
        return finalize("cost cap", meta={"spend": 2.14})

    session = _Session(_steps(), polls=3)
    gov = _run(session, judges=[cost_cap])
    assert _corrections(gov)[0]["meta"] == {"spend": 2.14}


def test_judges_receive_a_context_with_trajectory_and_elapsed():
    seen = {}

    @judge(name="probe")
    def probe(ctx):
        seen["len"] = len(ctx)
        seen["elapsed"] = ctx.elapsed_s
        seen["memory"] = dict(ctx.memory)
        return None

    session = _Session(_steps(4), polls=3)
    _run(session, judges=[probe])
    assert seen["len"] == 4
    assert seen["elapsed"] >= 0.0
    assert seen["memory"] == {}


def test_passing_both_judge_and_judges_is_rejected():
    @judge(name="x")
    def x(ctx):
        return None

    with pytest.raises(ValueError):
        Governor(ActorRef("governor", "local"), judge=x, judges=[x])


def test_defaults_to_the_heuristic_judge():
    gov = Governor(ActorRef("governor", "local"))
    assert [j.name for j in gov._judges] == ["heuristic"]


def test_singular_judge_kwarg_still_works():
    @judge(name="solo")
    def solo(ctx):
        return redirect("only me")

    session = _Session(_steps(), polls=3)
    _run(session, judge=solo)
    assert [c.flag for c in session._pushed] == [CorrectionFlag.REDIRECT]


def test_latency_judge_fires_on_a_stalled_trajectory():
    # The trajectory never grows, so only the time-based re-judgment trigger
    # can get the latency judge asked at all after the first poll — and it must
    # still be asked, and still fire, despite the identical-verdict suppression.
    class _Counting:
        name = "latency_budget"

        def __init__(self, inner):
            self._inner = inner
            self.calls = 0

        def judge(self, ctx):
            self.calls += 1
            return self._inner.judge(ctx)

    counting = _Counting(LatencyBudgetJudge(max_seconds=0.0))
    session = _Session(_steps(1), polls=40)
    _run(session, judges=[counting], rejudge_interval=0.0)
    assert counting.calls > 1, "the stale trigger should re-ask a time-based judge"
    assert [c.flag for c in session._pushed] == [CorrectionFlag.FORCE_FINALIZE]


def test_latency_judge_pushes_once_across_a_multi_second_stall():
    # Regression guard: the latency judge used to interpolate the LIVE elapsed
    # reading into its note. Because the de-dup fingerprint is
    # (flag, note, step_index), a ticking clock made every repeat look like a
    # fresh verdict, so a stalled run collected one FORCE_FINALIZE per second.
    # The note must stay stable across the stall; the precise reading rides in
    # meta, which is deliberately not part of the fingerprint.
    session = _Session(_steps(1), polls=2500)   # ~2.5s at poll_interval=0.001
    _run(session, judges=[LatencyBudgetJudge(max_seconds=0.5)],
         rejudge_interval=0.1)                  # re-judged ~20x past the budget

    assert [c.flag for c in session._pushed] == [CorrectionFlag.FORCE_FINALIZE]
    assert len({c.note for c in session._pushed}) == 1
    # meta still carries the real elapsed time, so nothing is lost by the fix.
    assert session._pushed[0].meta["elapsed_s"] > 0.5


def test_a_changed_verdict_from_the_same_judge_still_gets_through():
    # De-duplication suppresses only an IDENTICAL repeat; escalation must pass.
    notes = iter(["first", "second", "third"])

    @judge(name="escalating")
    def escalating(ctx):
        return redirect(next(notes, "third"))

    session = _Session(_steps(1), polls=40)
    _run(session, judges=[escalating], rejudge_interval=0.0)
    assert [c.note for c in session._pushed][:3] == ["first", "second", "third"]


def test_a_non_correction_return_is_logged_once_per_judge(caplog):
    @judge(name="bogus")
    def bogus(ctx):
        return "not a correction"

    session = _Session(_steps(1), polls=40)
    with caplog.at_level("WARNING", logger="sentinal.actors.governor"):
        _run(session, judges=[bogus], rejudge_interval=0.0)
    warnings = [r for r in caplog.records if "expected Correction" in r.message]
    assert len(warnings) == 1, f"logged {len(warnings)} times, expected once"


def test_dedup_memory_is_reset_per_session():
    @judge(name="steady")
    def steady(ctx):
        return redirect("same nudge")

    gov = Governor(ActorRef("governor", "local"), judges=[steady])
    s1 = _Session(_steps(1), polls=3)
    asyncio.run(gov.govern_session(s1, poll_interval=0.001))
    s2 = _Session(_steps(1), polls=3)
    asyncio.run(gov.govern_session(s2, poll_interval=0.001))
    assert len(s1._pushed) == 1
    assert len(s2._pushed) == 1, "session 2 must be judged from a clean slate"


def test_unchanged_trajectory_is_not_re_judged_within_the_interval():
    calls = {"n": 0}

    @judge(name="counter")
    def counter(ctx):
        calls["n"] += 1
        return None

    session = _Session(_steps(1), polls=8)
    _run(session, judges=[counter], rejudge_interval=1000.0)
    assert calls["n"] == 1


def test_class_and_function_judges_compose_in_one_list():
    @judge(name="fn")
    def fn(ctx):
        return None

    session = _Session(_steps(3), polls=3)
    gov = _run(session, judges=[RepeatToolJudge(threshold=3), fn])
    assert [c["judge"] for c in _corrections(gov)] == ["repeat_tool"]
    assert session._pushed[0].flag is CorrectionFlag.AVOID_TOOL
