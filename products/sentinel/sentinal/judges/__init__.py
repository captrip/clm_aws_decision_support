"""Trajectory judges: a curated built-in set plus the @judge decorator.

Judges are composed explicitly, never auto-registered:

    runtime.execute(session, judges=[StepBudgetJudge(), my_cost_judge])

When several judges fire on one step, the Governor pushes the most severe
correction and records them all.
"""
from sentinal.judges.builtin import (
    HeuristicJudge,
    LatencyBudgetJudge,
    OutputLoopJudge,
    RepeatToolJudge,
    StepBudgetJudge,
)
from sentinal.judges.decorator import (
    FunctionJudge,
    avoid_tool,
    finalize,
    judge,
    redirect,
    stop,
)

__all__ = [
    "FunctionJudge",
    "judge",
    "redirect",
    "avoid_tool",
    "finalize",
    "stop",
    "StepBudgetJudge",
    "RepeatToolJudge",
    "OutputLoopJudge",
    "LatencyBudgetJudge",
    "HeuristicJudge",
]
