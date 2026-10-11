"""The curated built-in judges: the four failure modes an agentic loop has.

Each is deterministic — same context in, same correction out — and each returns
None when it has nothing to say.
"""
from __future__ import annotations

from typing import Sequence

from sentinal.core.trajectory import Correction, JudgeContext, TrajectoryStep
from sentinal.judges.decorator import avoid_tool, finalize, redirect


class StepBudgetJudge:
    """Runaway ReAct loop: too many steps. -> FORCE_FINALIZE"""

    name = "step_budget"

    def __init__(self, max_steps: int = 20) -> None:
        self._max_steps = max_steps

    def judge(self, ctx: JudgeContext) -> Correction | None:
        n = len(ctx)
        if n > self._max_steps:
            return finalize("step budget exceeded", step_index=n - 1,
                            meta={"steps": n, "max_steps": self._max_steps})
        return None


class RepeatToolJudge:
    """The agent keeps calling the same tool. -> AVOID_TOOL"""

    name = "repeat_tool"

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold

    def judge(self, ctx: JudgeContext) -> Correction | None:
        n = len(ctx)
        if n < self._threshold:
            return None
        tail = ctx[-self._threshold:]
        if all(s.kind == "tool" and s.summary == tail[0].summary for s in tail):
            return avoid_tool(tail[0].summary, step_index=n - 1,
                              meta={"tool": tail[0].summary,
                                    "repeats": self._threshold})
        return None


class OutputLoopJudge:
    """The model is repeating its own output. -> REDIRECT"""

    name = "output_loop"

    def __init__(self, window: int = 3) -> None:
        self._window = window

    def judge(self, ctx: JudgeContext) -> Correction | None:
        n = len(ctx)
        if n < self._window:
            return None
        tail = ctx[-self._window:]
        if all(s.kind == "agent" and s.summary == tail[0].summary for s in tail):
            return redirect("repeated output; change approach", step_index=n - 1,
                            meta={"repeats": self._window})
        return None


class LatencyBudgetJudge:
    """Wall-clock cap. Fires even on a stalled trajectory. -> FORCE_FINALIZE"""

    name = "latency_budget"

    def __init__(self, max_seconds: float) -> None:
        self._max_seconds = max_seconds

    def judge(self, ctx: JudgeContext) -> Correction | None:
        if ctx.elapsed_s > self._max_seconds:
            # The note is deliberately free of the live elapsed value. A stalled
            # trajectory is re-judged every rejudge_interval, and the Governor
            # de-duplicates a judge's repeat verdict on (flag, note, step_index);
            # interpolating a ticking clock would defeat that and push one
            # FORCE_FINALIZE per second. The exact reading rides in meta.
            return finalize(
                f"latency budget exceeded ({self._max_seconds:.1f}s)",
                step_index=len(ctx) - 1,
                meta={"elapsed_s": ctx.elapsed_s,
                      "max_seconds": self._max_seconds})
        return None


class HeuristicJudge:
    """Backwards-compatible alias composing the three deterministic judges, in
    the original rule order: step budget, then repeated tool, then output loop.

    Prefer composing the individual judges:
        judges=[StepBudgetJudge(20), RepeatToolJudge(3), OutputLoopJudge(3)]
    """

    name = "heuristic"

    def __init__(self, *, repeat_threshold: int = 3, step_budget: int = 20,
                 loop_window: int = 3) -> None:
        self._judges = (
            StepBudgetJudge(max_steps=step_budget),
            RepeatToolJudge(threshold=repeat_threshold),
            OutputLoopJudge(window=loop_window),
        )

    def judge(
        self, ctx: JudgeContext | Sequence[TrajectoryStep]
    ) -> Correction | None:
        # Accept a bare list for callers predating JudgeContext.
        if not isinstance(ctx, JudgeContext):
            ctx = JudgeContext(trajectory=list(ctx))
        for j in self._judges:
            c = j.judge(ctx)
            if c is not None:
                return c
        return None
