from sentinal.core.trajectory import CorrectionFlag, JudgeContext, TrajectoryStep
from sentinal.judges import (
    HeuristicJudge, LatencyBudgetJudge, OutputLoopJudge, RepeatToolJudge,
    StepBudgetJudge,
)


def _step(i, kind, summary):
    return TrajectoryStep(index=i, kind=kind, summary=summary, raw={})


def _ctx(steps, **kw):
    return JudgeContext(trajectory=steps, **kw)


# ── StepBudgetJudge ──────────────────────────────────────────────────────────

def test_step_budget_is_quiet_under_budget():
    assert StepBudgetJudge(max_steps=20).judge(
        _ctx([_step(i, "agent", f"s{i}") for i in range(20)])) is None


def test_step_budget_finalizes_over_budget():
    c = StepBudgetJudge(max_steps=20).judge(
        _ctx([_step(i, "agent", f"s{i}") for i in range(21)]))
    assert c.flag is CorrectionFlag.FORCE_FINALIZE
    assert c.meta["steps"] == 21
    assert c.meta["max_steps"] == 20


def test_step_budget_is_quiet_on_empty_trajectory():
    assert StepBudgetJudge().judge(_ctx([])) is None


# ── RepeatToolJudge ──────────────────────────────────────────────────────────

def test_repeat_tool_fires_on_identical_consecutive_calls():
    c = RepeatToolJudge(threshold=3).judge(
        _ctx([_step(i, "tool", "web_search") for i in range(3)]))
    assert c.flag is CorrectionFlag.AVOID_TOOL
    assert c.note == "web_search"
    assert c.meta["tool"] == "web_search"


def test_repeat_tool_quiet_when_tools_differ():
    steps = [_step(0, "tool", "a"), _step(1, "tool", "b"), _step(2, "tool", "a")]
    assert RepeatToolJudge(threshold=3).judge(_ctx(steps)) is None


def test_repeat_tool_quiet_below_threshold():
    steps = [_step(i, "tool", "web_search") for i in range(2)]
    assert RepeatToolJudge(threshold=3).judge(_ctx(steps)) is None


def test_repeat_tool_ignores_agent_steps():
    steps = [_step(i, "agent", "same") for i in range(3)]
    assert RepeatToolJudge(threshold=3).judge(_ctx(steps)) is None


# ── OutputLoopJudge ──────────────────────────────────────────────────────────

def test_output_loop_fires_on_repeated_agent_output():
    c = OutputLoopJudge(window=3).judge(
        _ctx([_step(i, "agent", "same") for i in range(3)]))
    assert c.flag is CorrectionFlag.REDIRECT
    assert "approach" in c.note


def test_output_loop_quiet_when_output_changes():
    steps = [_step(i, "agent", f"different-{i}") for i in range(3)]
    assert OutputLoopJudge(window=3).judge(_ctx(steps)) is None


def test_output_loop_ignores_tool_steps():
    steps = [_step(i, "tool", "web_search") for i in range(3)]
    assert OutputLoopJudge(window=3).judge(_ctx(steps)) is None


# ── LatencyBudgetJudge ───────────────────────────────────────────────────────

def test_latency_budget_quiet_under_cap():
    assert LatencyBudgetJudge(max_seconds=30).judge(
        _ctx([_step(0, "agent", "x")], elapsed_s=5.0)) is None


def test_latency_budget_finalizes_over_cap():
    c = LatencyBudgetJudge(max_seconds=30).judge(
        _ctx([_step(0, "agent", "x")], elapsed_s=31.0))
    assert c.flag is CorrectionFlag.FORCE_FINALIZE
    assert c.meta["elapsed_s"] == 31.0
    assert c.meta["max_seconds"] == 30


def test_latency_budget_fires_on_a_stalled_trajectory():
    # No new steps at all — this is the case the trajectory-growth gate misses.
    c = LatencyBudgetJudge(max_seconds=1).judge(_ctx([], elapsed_s=99.0))
    assert c.flag is CorrectionFlag.FORCE_FINALIZE


# ── HeuristicJudge alias ─────────────────────────────────────────────────────

def test_heuristic_judge_still_accepts_a_bare_list():
    # Backwards compatibility: the pre-JudgeContext call shape.
    traj = [_step(i, "tool", "web_search") for i in range(3)]
    assert HeuristicJudge().judge(traj).flag is CorrectionFlag.AVOID_TOOL


def test_heuristic_judge_matches_its_component_judges():
    # Compare the WHOLE Correction, not just the flag: note, step_index and meta
    # are part of the decomposition contract too.
    over_budget = [_step(i, "agent", f"s{i}") for i in range(21)]
    assert (HeuristicJudge(step_budget=20).judge(_ctx(over_budget))
            == StepBudgetJudge(max_steps=20).judge(_ctx(over_budget)))

    repeated = [_step(i, "tool", "web_search") for i in range(3)]
    assert (HeuristicJudge(repeat_threshold=3).judge(_ctx(repeated))
            == RepeatToolJudge(threshold=3).judge(_ctx(repeated)))

    looped = [_step(i, "agent", "same") for i in range(3)]
    assert (HeuristicJudge(repeat_threshold=99, loop_window=3)
            .judge(_ctx(looped))
            == OutputLoopJudge(window=3).judge(_ctx(looped)))


def test_heuristic_judge_gives_the_same_verdict_for_a_list_and_a_context():
    # The sequence-like backwards-compatibility mechanism must survive freezing
    # the trajectory: a bare list and a JudgeContext are interchangeable.
    for traj in ([_step(i, "tool", "web_search") for i in range(3)],
                 [_step(i, "agent", f"s{i}") for i in range(21)],
                 [_step(i, "agent", "same") for i in range(3)],
                 [_step(0, "agent", "solo")]):
        assert HeuristicJudge().judge(list(traj)) == HeuristicJudge().judge(_ctx(traj))


def test_repeat_tool_tail_slice_works_against_a_frozen_trajectory():
    ctx = _ctx([_step(i, "tool", "web_search") for i in range(5)])
    assert isinstance(ctx.trajectory, tuple)
    tail = ctx[-3:]
    assert [s.index for s in tail] == [2, 3, 4]
    assert tail[0].summary == "web_search"
    assert RepeatToolJudge(threshold=3).judge(ctx).note == "web_search"


def test_heuristic_judge_prefers_step_budget_over_repeat():
    # Preserves the original rule order: budget check came first.
    traj = [_step(i, "tool", "web_search") for i in range(21)]
    assert HeuristicJudge(step_budget=20, repeat_threshold=3).judge(
        _ctx(traj)).flag is CorrectionFlag.FORCE_FINALIZE


def test_builtins_expose_names():
    assert StepBudgetJudge().name == "step_budget"
    assert RepeatToolJudge().name == "repeat_tool"
    assert OutputLoopJudge().name == "output_loop"
    assert LatencyBudgetJudge(30).name == "latency_budget"
