from sentinal.core.trajectory import (
    Correction, CorrectionFlag, JudgeContext, TrajectoryJudge, TrajectoryStep,
)
from sentinal.judges import avoid_tool, finalize, judge, redirect, stop


def _ctx(n=1, **kw):
    steps = [TrajectoryStep(index=i, kind="tool", summary="s", raw={})
             for i in range(n)]
    return JudgeContext(trajectory=steps, **kw)


def test_constructors_produce_the_right_flags():
    assert redirect("try X").flag is CorrectionFlag.REDIRECT
    assert avoid_tool("web_search").flag is CorrectionFlag.AVOID_TOOL
    assert finalize("done").flag is CorrectionFlag.FORCE_FINALIZE
    assert stop("halt").flag is CorrectionFlag.STOP


def test_constructors_carry_note_meta_and_step_index():
    c = finalize("cost cap", meta={"spend": 2.14}, step_index=7)
    assert c.note == "cost cap"
    assert c.meta == {"spend": 2.14}
    assert c.step_index == 7


def test_avoid_tool_puts_the_tool_name_in_note():
    # The actor-side ladder reads note as the tool name.
    assert avoid_tool("web_search").note == "web_search"


def test_bare_decorator_produces_a_judge():
    @judge
    def always_redirect(ctx):
        return redirect("go left")

    assert isinstance(always_redirect, TrajectoryJudge)
    assert always_redirect.judge(_ctx()).flag is CorrectionFlag.REDIRECT


def test_decorator_with_name_sets_the_judge_name():
    @judge(name="cost_cap")
    def f(ctx):
        return None

    assert f.name == "cost_cap"


def test_decorator_defaults_name_to_the_function_name():
    @judge
    def my_metric(ctx):
        return None

    assert my_metric.name == "my_metric"


def test_decorated_function_is_still_directly_callable():
    @judge
    def f(ctx):
        return redirect(f"saw {len(ctx)} steps")

    assert f(_ctx(3)).note == "saw 3 steps"


def test_decorated_judge_may_return_none():
    @judge
    def quiet(ctx):
        return None

    assert quiet.judge(_ctx()) is None


def test_judge_reads_business_data_from_context_memory():
    @judge(name="cost_cap")
    def cost_cap(ctx):
        spend = ctx.memory.get("usd", 0.0)
        if spend > 2.0:
            return finalize(f"cost cap exceeded: ${spend:.2f}",
                            meta={"spend": spend})
        return None

    assert cost_cap.judge(_ctx(memory={"usd": 0.5})) is None
    c = cost_cap.judge(_ctx(memory={"usd": 2.5}))
    assert c.flag is CorrectionFlag.FORCE_FINALIZE
    assert c.meta == {"spend": 2.5}


def test_decorator_preserves_docstring():
    @judge
    def documented(ctx):
        """Explains itself."""
        return None

    assert documented.__doc__ == "Explains itself."
