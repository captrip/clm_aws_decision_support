from dataclasses import FrozenInstanceError

from sentinal.core.trajectory import (
    Correction, CorrectionFlag, JudgeContext, SEVERITY, TrajectoryStep,
)


def _steps(n, kind="tool", summary="web_search"):
    return [TrajectoryStep(index=i, kind=kind, summary=summary, raw={})
            for i in range(n)]


def test_correction_meta_defaults_to_empty_and_accepts_a_dict():
    assert Correction(flag=CorrectionFlag.REDIRECT).meta == {}
    c = Correction(flag=CorrectionFlag.STOP, meta={"spend": 2.14})
    assert c.meta["spend"] == 2.14


def test_corrections_with_equal_meta_are_equal():
    # Guards test_judge_is_pure: two judge() calls must compare equal.
    a = Correction(flag=CorrectionFlag.REDIRECT, meta={"k": 1})
    b = Correction(flag=CorrectionFlag.REDIRECT, meta={"k": 1})
    assert a == b


def test_correction_meta_is_read_only():
    c = Correction(flag=CorrectionFlag.STOP, meta={"spend": 2.14})
    assert c.meta["spend"] == 2.14
    try:
        c.meta["spend"] = 99.0
        assert False, "expected meta to be read-only"
    except TypeError:
        pass


def test_correction_meta_is_defensive_copy():
    # External mutation of the source dict must not leak into the Correction.
    m = {"spend": 1.0}
    c = Correction(flag=CorrectionFlag.STOP, meta=m)
    assert c.meta["spend"] == 1.0
    m["spend"] = 999.0
    assert c.meta["spend"] == 1.0  # unchanged


def test_judge_context_is_sequence_like():
    ctx = JudgeContext(trajectory=_steps(3))
    assert len(ctx) == 3
    assert ctx[-1].index == 2
    assert [s.index for s in ctx] == [0, 1, 2]


def test_judge_context_supports_slicing():
    # HeuristicJudge does trajectory[-self._repeat_threshold:]
    ctx = JudgeContext(trajectory=_steps(5))
    tail = ctx[-3:]
    assert [s.index for s in tail] == [2, 3, 4]


def test_judge_context_defaults():
    ctx = JudgeContext(trajectory=[])
    assert ctx.elapsed_s == 0.0
    assert ctx.memory == {}
    assert ctx.input == {}


def test_judge_context_memory_is_read_only():
    ctx = JudgeContext(trajectory=[], memory={"usd": 1.0})
    assert ctx.memory["usd"] == 1.0
    try:
        ctx.memory["usd"] = 99.0
        assert False, "expected memory to be read-only"
    except TypeError:
        pass


def test_judge_context_trajectory_is_immutable():
    ctx = JudgeContext(trajectory=_steps(3))
    for mutate in (lambda: ctx.trajectory.clear(),
                   lambda: ctx.trajectory.append(_steps(1)[0]),
                   lambda: ctx.trajectory.__setitem__(0, None)):
        try:
            mutate()
            assert False, "expected the trajectory to be immutable"
        except AttributeError:
            pass
        except TypeError:
            pass
    assert len(ctx) == 3


def test_judge_context_trajectory_is_a_defensive_copy():
    # A judge must not see the Governor's list change under it, and mutating
    # the source list after construction must not reach the context.
    steps = _steps(3)
    ctx = JudgeContext(trajectory=steps)
    steps.clear()
    assert len(ctx) == 3


def test_judge_context_is_frozen():
    ctx = JudgeContext(trajectory=[])
    try:
        ctx.elapsed_s = 5.0
        assert False, "expected FrozenInstanceError"
    except FrozenInstanceError:
        pass


def test_severity_orders_flags_by_escalation():
    assert (SEVERITY[CorrectionFlag.CONTINUE]
            < SEVERITY[CorrectionFlag.REDIRECT]
            < SEVERITY[CorrectionFlag.AVOID_TOOL]
            < SEVERITY[CorrectionFlag.FORCE_FINALIZE]
            < SEVERITY[CorrectionFlag.STOP])


def test_severity_covers_every_flag():
    assert set(SEVERITY) == set(CorrectionFlag)
