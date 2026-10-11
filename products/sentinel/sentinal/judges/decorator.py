"""Author a judge from a plain function, plus constructors for the four verbs a
judge can speak (CONTINUE is expressed by returning None).

    @judge(name="cost_cap")
    def cost_cap(ctx):
        spend = ctx.memory.get("usd", 0.0)
        if spend > 2.0:
            return finalize(f"cost cap exceeded: ${spend:.2f}",
                            meta={"spend": spend})
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Mapping

from sentinal.core.trajectory import (
    Correction, CorrectionFlag, JudgeContext,
)


def _make(flag: CorrectionFlag, note: str | None,
          meta: Mapping[str, Any] | None, step_index: int) -> Correction:
    return Correction(flag=flag, note=note, step_index=step_index,
                      meta=dict(meta or {}))


def redirect(note: str | None = None, *, meta: Mapping[str, Any] | None = None,
             step_index: int = -1) -> Correction:
    """Steer the agent; `note` is the guidance it will be shown."""
    return _make(CorrectionFlag.REDIRECT, note, meta, step_index)


def avoid_tool(tool: str, *, meta: Mapping[str, Any] | None = None,
               step_index: int = -1) -> Correction:
    """Stop the agent calling `tool`. The tool name travels in `note`."""
    return _make(CorrectionFlag.AVOID_TOOL, tool, meta, step_index)


def finalize(note: str | None = None, *, meta: Mapping[str, Any] | None = None,
             step_index: int = -1) -> Correction:
    """Tell the agent to wrap up now with what it already has."""
    return _make(CorrectionFlag.FORCE_FINALIZE, note, meta, step_index)


def stop(note: str | None = None, *, meta: Mapping[str, Any] | None = None,
         step_index: int = -1) -> Correction:
    """Abort the run."""
    return _make(CorrectionFlag.STOP, note, meta, step_index)


class FunctionJudge:
    """A judge backed by a plain function. Stays callable as the original
    function so it can be unit-tested without a Governor."""

    def __init__(self, fn: Callable[[JudgeContext], Correction | None],
                 name: str | None = None):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "judge")
        functools.update_wrapper(self, fn)

    def judge(self, ctx: JudgeContext) -> Correction | None:
        return self._fn(ctx)

    def __call__(self, ctx: JudgeContext) -> Correction | None:
        return self._fn(ctx)

    def __repr__(self) -> str:
        return f"<judge {self.name}>"


def judge(fn: Callable | None = None, *, name: str | None = None):
    """Turn a function into a TrajectoryJudge.

    Usage:
        @judge
        def my_metric(ctx): ...

        @judge(name="cost_cap")
        def cost_cap(ctx): ...

    The judge is NOT auto-registered — attach it explicitly:
        runtime.execute(session, judges=[cost_cap])
    """
    def decorator(func: Callable) -> FunctionJudge:
        return FunctionJudge(func, name=name)

    if fn is not None:
        return decorator(fn)
    return decorator
