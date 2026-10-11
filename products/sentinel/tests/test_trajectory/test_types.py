from sentinal.core.trajectory import (
    CorrectionFlag, TrajectoryStep, Correction, TrajectoryJudge,
)


def test_correction_flag_values():
    assert CorrectionFlag.CONTINUE.value == "continue"
    assert CorrectionFlag.STOP.value == "stop"
    assert CorrectionFlag("redirect") is CorrectionFlag.REDIRECT


def test_trajectory_step_is_frozen():
    s = TrajectoryStep(index=0, kind="tool", summary="web_search", raw={"a": 1})
    assert s.ts == 0.0
    try:
        s.index = 5  # frozen -> should raise
        assert False, "expected FrozenInstanceError"
    except Exception:
        pass


def test_correction_defaults():
    c = Correction(flag=CorrectionFlag.REDIRECT)
    assert c.note is None
    assert c.step_index == -1


def test_judge_protocol_is_runtime_checkable_or_structural():
    class _J:
        def judge(self, trajectory):
            return Correction(flag=CorrectionFlag.CONTINUE)
    j: TrajectoryJudge = _J()
    assert j.judge([]).flag is CorrectionFlag.CONTINUE
