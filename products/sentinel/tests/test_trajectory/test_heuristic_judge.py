from sentinal.core.trajectory import CorrectionFlag, TrajectoryStep
from sentinal.actors.judges import HeuristicJudge


def _step(i, kind, summary):
    return TrajectoryStep(index=i, kind=kind, summary=summary, raw={})


def test_empty_trajectory_is_quiet():
    assert HeuristicJudge().judge([]) is None


def test_healthy_trajectory_is_quiet():
    traj = [_step(0, "agent", "think"), _step(1, "tool", "web_search"),
            _step(2, "agent", "answer")]
    assert HeuristicJudge().judge(traj) is None


def test_repeated_tool_triggers_avoid_tool():
    traj = [_step(i, "tool", "web_search") for i in range(3)]
    c = HeuristicJudge(repeat_threshold=3).judge(traj)
    assert c.flag is CorrectionFlag.AVOID_TOOL
    assert c.note == "web_search"


def test_step_budget_triggers_force_finalize():
    traj = [_step(i, "agent", f"s{i}") for i in range(21)]
    c = HeuristicJudge(step_budget=20).judge(traj)
    assert c.flag is CorrectionFlag.FORCE_FINALIZE


def test_agent_loop_triggers_redirect():
    traj = [_step(i, "agent", "same") for i in range(3)]
    c = HeuristicJudge(repeat_threshold=99, loop_window=3).judge(traj)
    assert c.flag is CorrectionFlag.REDIRECT
    assert c.note and "approach" in c.note


def test_judge_is_pure():
    traj = [_step(i, "tool", "web_search") for i in range(3)]
    j = HeuristicJudge()
    assert j.judge(traj) == j.judge(traj)
