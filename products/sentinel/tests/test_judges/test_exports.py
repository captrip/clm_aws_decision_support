import sentinal


def test_public_judge_api_is_importable_from_the_package_root():
    names = [
        "judge", "FunctionJudge", "redirect", "avoid_tool", "finalize", "stop",
        "JudgeContext", "SEVERITY",
        "StepBudgetJudge", "RepeatToolJudge", "OutputLoopJudge",
        "LatencyBudgetJudge", "HeuristicJudge",
    ]
    for n in names:
        assert hasattr(sentinal, n), f"sentinal.{n} is missing"
        assert n in sentinal.__all__, f"{n} missing from sentinal.__all__"


def test_every_name_in_sentinal_judges_all_is_re_exported_from_the_root():
    import sentinal.judges as judges_pkg
    for n in judges_pkg.__all__:
        assert hasattr(sentinal, n), f"sentinal.{n} is missing"
        assert n in sentinal.__all__, f"{n} missing from sentinal.__all__"


def test_function_judge_is_usable_for_isinstance_from_the_root():
    from sentinal import FunctionJudge, judge

    @judge(name="x")
    def x(ctx):
        return None

    assert isinstance(x, FunctionJudge)


def test_a_judge_can_be_written_using_only_root_imports():
    from sentinal import finalize, judge

    @judge(name="cost_cap")
    def cost_cap(ctx):
        spend = ctx.memory.get("usd", 0.0)
        if spend > 2.0:
            return finalize(f"cost cap: ${spend:.2f}", meta={"spend": spend})
        return None

    from sentinal import JudgeContext
    assert cost_cap.judge(JudgeContext(trajectory=[], memory={"usd": 3.0}))
    assert cost_cap.judge(JudgeContext(trajectory=[], memory={"usd": 1.0})) is None
