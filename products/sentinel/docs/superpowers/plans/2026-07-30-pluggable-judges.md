# Pluggable Judges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users compose several trajectory judges on one session — a curated built-in set plus their own business-metric judges — without changing the correction vocabulary or any executable's correction-handling code.

**Architecture:** Three layers. (1) Types in `sentinal/core/trajectory.py` gain `Correction.meta` for structured business detail and a sequence-like `JudgeContext` that carries wall-clock and session memory while staying backwards-compatible with judges written against `list[TrajectoryStep]`. (2) A new `sentinal/judges/` package holds the four built-in judges, four flag constructors, and the `@judge` decorator that turns a plain function into a judge. (3) `Governor` runs N judges per step, records every finding, and pushes only the highest-severity correction.

**Tech Stack:** Python 3.13+ (running on 3.14 free-threaded), stdlib only (`dataclasses`, `typing.Protocol`, `types.MappingProxyType`, `time.monotonic`), pytest with `asyncio_mode = "auto"`.

**Spec:** `docs/superpowers/specs/2026-07-30-pluggable-judges-design.md`

## Global Constraints

- **Python:** `requires-python = ">=3.13"`. No new runtime dependencies — `dependencies = []` in `pyproject.toml` must stay empty.
- **Test runner:** `PYTHON_GIL=0 .ft-env/bin/python -m pytest`. The venv is Python 3.14 free-threaded (`python3.14t`). **The `PYTHON_GIL=0` prefix is mandatory** — `tests/test_gpu_parallelism.py` and `tests/test_compute/test_e2e_parallelism.py` have an autouse fixture asserting `sys._is_gil_enabled() is False`, so without it 5 tests fail and 14 error before your code is even reached.
- **Verified baselines** (measured 2026-07-30, before any of this work):
  - `PYTHON_GIL=0 … pytest tests/test_trajectory/ -q` → **35 passed, 1 skipped**. This is the suite your tasks affect; it must stay green.
  - `PYTHON_GIL=0 … pytest tests/ -q` → **2 failed, 151 passed, 1 skipped**. The two failures are **pre-existing and unrelated** GPU/timing benchmarks: `test_matmul_gpu_faster_than_cpu[8192]` (measured 1.696x vs a >1.7x threshold) and `test_runtime_wall_time_shows_concurrency` (passes in isolation; flaky only under full-suite load). Do not attempt to fix these, and do not treat them as regressions from your task.
- **No breaking changes.** Every existing test must keep passing untouched. Specifically: `tests/test_trajectory/test_heuristic_judge.py` (5 tests calling `HeuristicJudge().judge(list)`), `tests/test_trajectory/test_types.py` (calls `_J().judge([])`), `tests/test_trajectory/test_governor_corrections.py` (fake judges with signature `judge(self, trajectory)` and fake sessions with no `.context` attribute), `tests/test_trajectory/test_integration.py`.
- **Correction vocabulary is closed.** Do not add members to `CorrectionFlag`. Business detail goes in `note` (str) and `meta` (dict).
- **Judges must not reach the session.** A judge receives `JudgeContext` only. Never pass `ExecutionSession` to a judge.
- **A judge is never trusted.** A raising, hanging-free, or wrong-typed judge must never crash a run or suppress its peers.
- **Naming:** the built-in judge classes are exactly `StepBudgetJudge`, `RepeatToolJudge`, `OutputLoopJudge`, `LatencyBudgetJudge`. The decorator is `judge`. The constructors are `redirect`, `avoid_tool`, `finalize`, `stop`.
- **Commit style:** Conventional Commits (`feat:`, `test:`, `fix:`, `refactor:`), matching recent history.

## File Structure

**Created:**
- `sentinal/judges/__init__.py` — public surface of the judges package; re-exports the decorator, constructors, and built-ins.
- `sentinal/judges/decorator.py` — `judge()` decorator + the four flag constructors. No dependency on the built-ins.
- `sentinal/judges/builtin.py` — `StepBudgetJudge`, `RepeatToolJudge`, `OutputLoopJudge`, `LatencyBudgetJudge`, and the `HeuristicJudge` composing alias.
- `tests/test_judges/__init__.py`
- `tests/test_judges/test_decorator.py`
- `tests/test_judges/test_builtin.py`
- `tests/test_trajectory/test_judge_context.py`
- `tests/test_trajectory/test_governor_arbitration.py`

**Modified:**
- `sentinal/core/trajectory.py` — add `meta` to `Correction`, add `JudgeContext`, add `SEVERITY`, widen the `TrajectoryJudge` protocol.
- `sentinal/actors/governor.py:49-55` (constructor) and `:96-123` (judging loop) — multi-judge, arbitration, context construction, re-judgment trigger.
- `sentinal/actors/judges.py` — becomes a 3-line backwards-compatibility shim re-exporting `HeuristicJudge` from `sentinal.judges.builtin`.
- `sentinal/runtime/runtime.py:123-185` — `execute(session, judges=None)`.
- `sentinal/__init__.py` — export the new public names.
- `examples/search_agent.py` — show a composed `judges=[...]` list including a user-defined business judge; print which judge issued each correction.

**Why this split:** `decorator.py` and `builtin.py` are separated so the built-ins can be authored *using* the decorator's constructors without the decorator importing them (no cycle), and so a user reading "how do I write a judge" opens one small file. `sentinal/actors/judges.py` shrinks to a shim rather than being deleted, because `tests/test_trajectory/test_heuristic_judge.py` and `tests/test_trajectory/test_integration.py` import from it.

---

### Task 1: Types — `Correction.meta`, `JudgeContext`, `SEVERITY`

**Files:**
- Modify: `sentinal/core/trajectory.py` (whole file, currently 37 lines)
- Test: `tests/test_trajectory/test_judge_context.py` (create)

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `Correction(flag, note=None, step_index=-1, ts=0.0, meta={})` — `meta: Mapping[str, Any]`
  - `JudgeContext(trajectory, elapsed_s=0.0, memory={}, input={})` — frozen, sequence-like over `trajectory`
  - `SEVERITY: dict[CorrectionFlag, int]` — `CONTINUE:0, REDIRECT:1, AVOID_TOOL:2, FORCE_FINALIZE:3, STOP:4`
  - `TrajectoryJudge` protocol: `judge(self, ctx: JudgeContext) -> Correction | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trajectory/test_judge_context.py`:

```python
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


def test_judge_context_is_frozen():
    ctx = JudgeContext(trajectory=[])
    try:
        ctx.elapsed_s = 5.0
        assert False, "expected FrozenInstanceError"
    except Exception:
        pass


def test_severity_orders_flags_by_escalation():
    assert (SEVERITY[CorrectionFlag.CONTINUE]
            < SEVERITY[CorrectionFlag.REDIRECT]
            < SEVERITY[CorrectionFlag.AVOID_TOOL]
            < SEVERITY[CorrectionFlag.FORCE_FINALIZE]
            < SEVERITY[CorrectionFlag.STOP])


def test_severity_covers_every_flag():
    assert set(SEVERITY) == set(CorrectionFlag)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_judge_context.py -q`
Expected: FAIL — `ImportError: cannot import name 'JudgeContext' from 'sentinal.core.trajectory'`

- [ ] **Step 3: Rewrite `sentinal/core/trajectory.py`**

Replace the whole file with:

```python
"""Trajectory steps, judge input, and Governor correction flags for mid-flight steering."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


class CorrectionFlag(str, Enum):
    CONTINUE = "continue"            # all good, no-op (never pushed)
    REDIRECT = "redirect"            # steer: use note as guidance
    AVOID_TOOL = "avoid_tool"        # note = tool name to stop calling
    FORCE_FINALIZE = "force_finalize"  # wrap up now
    STOP = "stop"                    # abort the run


# How escalatory each flag is. When several judges disagree on one step, the
# Governor pushes the highest-severity correction and records the rest.
SEVERITY: dict[CorrectionFlag, int] = {
    CorrectionFlag.CONTINUE: 0,
    CorrectionFlag.REDIRECT: 1,
    CorrectionFlag.AVOID_TOOL: 2,
    CorrectionFlag.FORCE_FINALIZE: 3,
    CorrectionFlag.STOP: 4,
}


@dataclass(frozen=True)
class TrajectoryStep:
    index: int
    kind: str          # "agent" | "tool" | "message"
    summary: str       # short human-readable
    raw: dict          # streamed chunk, for the judge
    ts: float = 0.0


@dataclass(frozen=True)
class Correction:
    flag: CorrectionFlag
    note: str | None = None
    step_index: int = -1
    ts: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)
    """Free-form structured detail from the judge (e.g. {"spend": 2.14}).
    Never interpreted by the framework; carried to the interventions log and
    the dashboard so business metrics survive to the operator."""


def _read_only(m: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(m or {}))


@dataclass(frozen=True)
class JudgeContext:
    """What a judge sees. Sequence-like over `trajectory`, so judges written
    against `list[TrajectoryStep]` keep working unchanged: len(), indexing,
    slicing and iteration all delegate to the trajectory."""

    trajectory: list[TrajectoryStep]
    elapsed_s: float = 0.0
    """Wall-clock seconds since the Governor started governing this session."""
    memory: Mapping[str, Any] = field(default_factory=dict)
    """Read-only snapshot of session memory."""
    input: Mapping[str, Any] = field(default_factory=dict)
    """The execution input, e.g. {"query": "..."}."""

    def __post_init__(self):
        # Freeze the mappings so a judge cannot mutate session state.
        object.__setattr__(self, "memory", _read_only(self.memory))
        object.__setattr__(self, "input", _read_only(self.input))

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(self, i):
        return self.trajectory[i]

    def __iter__(self):
        return iter(self.trajectory)


@runtime_checkable
class TrajectoryJudge(Protocol):
    def judge(self, ctx: JudgeContext) -> Correction | None: ...
```

Two notes on this file:

- `JudgeContext` uses `@dataclass(frozen=True)` **without** `slots=True`. Do not add `slots=True` — `__post_init__` reassigns frozen fields via `object.__setattr__`, which is the standard escape hatch but interacts badly with slots on a frozen dataclass.
- Verified on the project's Python 3.14t: `MappingProxyType` raises `TypeError` on item assignment (what `test_judge_context_memory_is_read_only` asserts), and two `JudgeContext` values holding equal mappingproxies compare equal (so `test_judge_is_pure` in the existing suite still holds).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_judge_context.py -q`
Expected: 9 passed

- [ ] **Step 5: Verify no regressions**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/ -q`
Expected: 44 passed, 1 skipped (35+9). If `test_judge_is_pure` or `test_correction_defaults` fails, the `meta` default is wrong — it must be a `field(default_factory=dict)`, not a bare `{}`.

- [ ] **Step 6: Commit**

```bash
git add sentinal/core/trajectory.py tests/test_trajectory/test_judge_context.py
git commit -m "feat: add Correction.meta, JudgeContext, and flag severity ranking"
```

---

### Task 2: `@judge` decorator and flag constructors

**Files:**
- Create: `sentinal/judges/__init__.py`, `sentinal/judges/decorator.py`
- Create: `tests/test_judges/__init__.py`, `tests/test_judges/test_decorator.py`

**Interfaces:**
- Consumes: `Correction`, `CorrectionFlag`, `JudgeContext` from `sentinal.core.trajectory` (Task 1).
- Produces:
  - `redirect(note=None, *, meta=None, step_index=-1) -> Correction`
  - `avoid_tool(tool, *, meta=None, step_index=-1) -> Correction`
  - `finalize(note=None, *, meta=None, step_index=-1) -> Correction`
  - `stop(note=None, *, meta=None, step_index=-1) -> Correction`
  - `judge(fn=None, *, name=None) -> FunctionJudge | decorator` — supports both `@judge` and `@judge(name="x")`
  - `FunctionJudge` with `.judge(ctx)`, `.name`, and `__call__(ctx)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judges/__init__.py` (empty file), then `tests/test_judges/test_decorator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_judges/test_decorator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinal.judges'`

- [ ] **Step 3: Write `sentinal/judges/decorator.py`**

```python
"""Author a judge from a plain function, plus constructors for the five verbs.

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
```

`functools.update_wrapper` is what makes `test_decorator_preserves_docstring` pass — it copies `__doc__`, `__name__`, and `__wrapped__` onto the instance. It does not touch `self.name`, which is a separate attribute.

Verified on the project's Python 3.14t: a `FunctionJudge` instance satisfies `isinstance(obj, TrajectoryJudge)` against the `runtime_checkable` Protocol, so `@judge`-decorated functions are structurally judges.

- [ ] **Step 4: Write `sentinal/judges/__init__.py`**

Built-ins are added to this file in Task 3.

```python
"""Trajectory judges: a curated built-in set plus the @judge decorator.

Judges are composed explicitly, never auto-registered:

    runtime.execute(session, judges=[StepBudgetJudge(), my_cost_judge])

When several judges fire on one step, the Governor pushes the most severe
correction and records them all.
"""
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
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_judges/test_decorator.py -q`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add sentinal/judges/ tests/test_judges/
git commit -m "feat: add @judge decorator and correction flag constructors"
```

---

### Task 3: Built-in judges + `HeuristicJudge` alias

**Files:**
- Create: `sentinal/judges/builtin.py`
- Modify: `sentinal/judges/__init__.py` (add built-in exports)
- Modify: `sentinal/actors/judges.py` (whole file → shim)
- Create: `tests/test_judges/test_builtin.py`

**Interfaces:**
- Consumes: `redirect`, `avoid_tool`, `finalize` from `sentinal.judges.decorator` (Task 2); `JudgeContext`, `Correction` from `sentinal.core.trajectory` (Task 1).
- Produces:
  - `StepBudgetJudge(max_steps=20)` → `FORCE_FINALIZE` when `len(ctx) > max_steps`
  - `RepeatToolJudge(threshold=3)` → `AVOID_TOOL` when the last `threshold` steps are the same tool
  - `OutputLoopJudge(window=3)` → `REDIRECT` when the last `window` steps are the same agent output
  - `LatencyBudgetJudge(max_seconds)` → `FORCE_FINALIZE` when `ctx.elapsed_s > max_seconds`
  - `HeuristicJudge(repeat_threshold=3, step_budget=20, loop_window=3)` — composing alias, same verdicts as before
  - All expose `.name`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_judges/test_builtin.py`:

```python
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
    over_budget = [_step(i, "agent", f"s{i}") for i in range(21)]
    assert (HeuristicJudge(step_budget=20).judge(_ctx(over_budget)).flag
            is StepBudgetJudge(max_steps=20).judge(_ctx(over_budget)).flag)

    repeated = [_step(i, "tool", "web_search") for i in range(3)]
    assert (HeuristicJudge(repeat_threshold=3).judge(_ctx(repeated)).flag
            is RepeatToolJudge(threshold=3).judge(_ctx(repeated)).flag)

    looped = [_step(i, "agent", "same") for i in range(3)]
    assert (HeuristicJudge(repeat_threshold=99, loop_window=3)
            .judge(_ctx(looped)).flag
            is OutputLoopJudge(window=3).judge(_ctx(looped)).flag)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_judges/test_builtin.py -q`
Expected: FAIL — `ImportError: cannot import name 'HeuristicJudge' from 'sentinal.judges'`

- [ ] **Step 3: Write `sentinal/judges/builtin.py`**

```python
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
            return finalize(f"latency budget exceeded ({ctx.elapsed_s:.1f}s)",
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
```

Note `HeuristicJudge.judge` now returns `None` instead of `Correction(CONTINUE)` when nothing fires. The Governor treats both identically (Task 4), and `tests/test_trajectory/test_heuristic_judge.py` only asserts `.flag` on the *firing* cases — except `test_empty_trajectory_continues` and `test_healthy_trajectory_continues`, which assert `.flag is CONTINUE`. **Those two tests must be updated in Step 5** — this is the one intentional test change in the plan.

- [ ] **Step 4: Update `sentinal/judges/__init__.py`**

Add to the imports and `__all__`:

```python
from sentinal.judges.builtin import (
    HeuristicJudge,
    LatencyBudgetJudge,
    OutputLoopJudge,
    RepeatToolJudge,
    StepBudgetJudge,
)
```

and extend `__all__` with `"StepBudgetJudge", "RepeatToolJudge", "OutputLoopJudge", "LatencyBudgetJudge", "HeuristicJudge"`.

- [ ] **Step 5: Update the two `HeuristicJudge` CONTINUE tests**

In `tests/test_trajectory/test_heuristic_judge.py`, replace the two tests that assert a `CONTINUE` flag. A quiet judge now returns `None`:

```python
def test_empty_trajectory_is_quiet():
    assert HeuristicJudge().judge([]) is None


def test_healthy_trajectory_is_quiet():
    traj = [_step(0, "agent", "think"), _step(1, "tool", "web_search"),
            _step(2, "agent", "answer")]
    assert HeuristicJudge().judge(traj) is None
```

Leave the other three tests and `test_judge_is_pure` exactly as they are — they still pass (`None == None` holds for the purity test).

- [ ] **Step 6: Turn `sentinal/actors/judges.py` into a shim**

Replace the whole file (currently 42 lines) with:

```python
"""Backwards-compatible import location. The judges now live in sentinal.judges."""
from sentinal.judges.builtin import HeuristicJudge

__all__ = ["HeuristicJudge"]
```

- [ ] **Step 7: Run the tests**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_judges/ tests/test_trajectory/ -q`
Expected: all pass (63 passed, 1 skipped). If `tests/test_trajectory/test_integration.py` fails here, the rule ordering in `HeuristicJudge` is wrong — `StepBudgetJudge` must be checked first.

- [ ] **Step 8: Commit**

```bash
git add sentinal/judges/ sentinal/actors/judges.py tests/test_judges/ tests/test_trajectory/test_heuristic_judge.py
git commit -m "refactor: split HeuristicJudge into composable built-in judges"
```

---

### Task 4: Governor — N judges, severity arbitration, fault isolation

**Files:**
- Modify: `sentinal/actors/governor.py:49-55` (constructor), `:82-139` (`govern_session`)
- Create: `tests/test_trajectory/test_governor_arbitration.py`

**Interfaces:**
- Consumes: `SEVERITY`, `JudgeContext` (Task 1); `HeuristicJudge` (Task 3).
- Produces:
  - `Governor(ref, quality_threshold=0.7, judge=None, judges=None, rejudge_interval=1.0)`
  - `Governor.set_judges(judges: Sequence[TrajectoryJudge]) -> None`
  - `get_interventions()` correction entries gain keys: `judge` (str), `meta` (dict), `pushed` (bool)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trajectory/test_governor_arbitration.py`:

```python
import asyncio

import pytest

from sentinal.core.actor_ref import ActorRef
from sentinal.actors.governor import Governor
from sentinal.core.trajectory import CorrectionFlag, TrajectoryStep
from sentinal.judges import (
    LatencyBudgetJudge, RepeatToolJudge, finalize, judge, redirect, stop,
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


def test_a_judge_returning_a_non_correction_is_a_no_op():
    @judge(name="bogus")
    def bogus(ctx):
        return "not a correction"

    session = _Session(_steps(), polls=3)
    _run(session, judges=[bogus])   # must not raise
    assert session._pushed == []


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
    # can get the latency judge asked a second time.
    session = _Session(_steps(1), polls=40)
    _run(session, judges=[LatencyBudgetJudge(max_seconds=0.0)],
         rejudge_interval=0.0)
    assert len(session._pushed) > 1
    assert all(c.flag is CorrectionFlag.FORCE_FINALIZE for c in session._pushed)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_governor_arbitration.py -q`
Expected: FAIL — `TypeError: Governor.__init__() got an unexpected keyword argument 'judges'`

- [ ] **Step 3: Replace the Governor constructor**

In `sentinal/actors/governor.py`, replace lines 49-55:

```python
    def __init__(self, *args, quality_threshold: float = 0.7,
                 judge: TrajectoryJudge | None = None,
                 judges: Sequence[TrajectoryJudge] | None = None,
                 rejudge_interval: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if judge is not None and judges is not None:
            raise ValueError(
                "pass either judge= (one judge) or judges= (several), not both")
        self.quality_threshold = quality_threshold
        self._interventions = []
        self._judges: list[TrajectoryJudge] = (
            list(judges) if judges is not None
            else [judge] if judge is not None
            else [HeuristicJudge()]
        )
        self._rejudge_interval = rejudge_interval
        self._last_judged_len = 0
        self._last_judged_at = 0.0

    def set_judges(self, judges: Sequence[TrajectoryJudge]) -> None:
        """Replace this Governor's judges (used by the runtime's judges= kwarg)."""
        self._judges = list(judges)
```

Update the imports at the top of the file:

```python
from typing import TYPE_CHECKING, Sequence

from sentinal.actors.actor import BaseActor
from sentinal.actors.judges import HeuristicJudge
from sentinal.core.trajectory import (
    SEVERITY, Correction, CorrectionFlag, JudgeContext, TrajectoryJudge,
)
```

- [ ] **Step 4: Replace the judging loop**

Replace lines 96-123 of `govern_session` (from `self._last_judged_len = 0` through the end of the `if correction is not None...` block) with:

```python
        self._last_judged_len = 0
        self._last_judged_at = 0.0
        t0 = time.monotonic()

        sid = str(session.session_id)
        activity_bus.emit("Governor", "governing", status="governing", session_id=sid)

        while not session.is_terminal():
            trajectory = session.get_trajectory()
            elapsed = time.monotonic() - t0
            # Re-judge when the trajectory grew, or when enough time has passed
            # that a time-based judge deserves another look — without the second
            # condition a stalled trajectory is never judged again.
            grew = len(trajectory) > self._last_judged_len
            stale = (elapsed - self._last_judged_at) >= self._rejudge_interval
            if grew or stale:
                self._last_judged_len = len(trajectory)
                self._last_judged_at = elapsed
                self._judge_and_arbitrate(session, trajectory, elapsed)
```

Then add the two new methods to the class:

```python
    def _build_context(self, session, trajectory, elapsed: float) -> JudgeContext:
        """Assemble what judges are allowed to see. Defensive about the session
        shape: tests and adapters pass sessions without a .context."""
        try:
            memory = session.state.get_all_memory()
        except Exception:
            memory = {}
        context = getattr(session, "context", None)
        raw_input = getattr(context, "input", None)
        return JudgeContext(
            trajectory=trajectory,
            elapsed_s=elapsed,
            memory=memory if isinstance(memory, dict) else {},
            input=raw_input if isinstance(raw_input, dict) else {},
        )

    def _judge_and_arbitrate(self, session, trajectory, elapsed: float) -> None:
        """Run every judge, record every finding, push only the most severe."""
        ctx = self._build_context(session, trajectory, elapsed)

        verdicts: list[tuple[str, Correction]] = []
        for j in self._judges:
            name = getattr(j, "name", None) or type(j).__name__
            try:
                c = j.judge(ctx)
            except Exception as ex:   # a broken judge must not crash the run
                logger.warning(f"Governor judge {name!r} failed: {ex}")
                continue
            if c is None:
                continue
            if not isinstance(c, Correction):
                logger.warning(
                    f"Governor judge {name!r} returned {type(c).__name__}, "
                    "expected Correction or None — ignoring")
                continue
            if c.flag != CorrectionFlag.CONTINUE:
                verdicts.append((name, c))

        if not verdicts:
            return

        # Highest severity wins; ties break by registration order because max()
        # keeps the first maximal element.
        winner_idx = max(range(len(verdicts)),
                         key=lambda i: SEVERITY[verdicts[i][1].flag])

        for i, (name, c) in enumerate(verdicts):
            self._interventions.append({
                "type": "correction",
                "judge": name,
                "flag": c.flag.value,
                "note": c.note,
                "meta": dict(c.meta),
                "pushed": i == winner_idx,
            })

        name, winner = verdicts[winner_idx]
        session.push_correction(winner)
        activity_bus.emit(
            "Governor", "correction",
            status="governing",
            detail=f"{name}:{winner.flag.value}:{winner.note or ''}",
            session_id=str(session.session_id),
        )
```

Add `import time` to the imports at the top of the file.

- [ ] **Step 5: Run the new tests**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_governor_arbitration.py -q`
Expected: 14 passed

- [ ] **Step 6: Run the full suite for regressions**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/ -q`
Expected: all pass. Watch two in particular:
- `test_governor_does_not_re_push_on_unchanged_trajectory` — passes because 5 polls × 0.001s ≈ 5ms, far under the 1.0s default `rejudge_interval`.
- `test_governor_survives_judge_exception` — its `_Boom.judge` has no `name` attribute, so the `getattr(j, "name", None) or type(j).__name__` fallback in `_judge_and_arbitrate` supplies `"_Boom"`.

- [ ] **Step 7: Commit**

```bash
git add sentinal/actors/governor.py tests/test_trajectory/test_governor_arbitration.py
git commit -m "feat: Governor runs multiple judges with severity arbitration"
```

---

### Task 5: Runtime — `execute(session, judges=...)`

**Files:**
- Modify: `sentinal/runtime/runtime.py:123-185`
- Test: `tests/test_trajectory/test_integration.py` (append)

**Interfaces:**
- Consumes: `Governor(judges=...)` and `Governor.set_judges()` (Task 4).
- Produces: `SentinelRuntime.execute(session, judges=None) -> ExecutionSession`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trajectory/test_integration.py`:

```python
from sentinal.judges import judge, stop


class _CountingExecutable(Executable):
    """Emits steps until stopped; records where it stopped."""
    async def execute(self, context):
        session = context.session
        assert session is not None
        i = 0
        while i < 50:
            session.append_step(
                TrajectoryStep(index=i, kind="agent", summary=f"step-{i}", raw={})
            )
            for c in session.drain_corrections():
                if c.flag.value == "stop":
                    raise ExecutionAborted(c.note or "stop")
            await asyncio.sleep(0.01)
            i += 1
        return {"finalized_at": i}


async def _run_with_judges(judges):
    runtime = SentinelRuntime()
    session = ExecutionSession(
        executable=_CountingExecutable(),
        context=ExecutionContext(input={"query": "hello"}),
    )
    await runtime.execute(session, judges=judges)
    governor = next(a for a in runtime.actors if a.ref.name == "governor")
    interventions = governor.get_interventions()
    await runtime.shutdown()
    return session, interventions


def test_custom_business_judge_attached_via_judges_kwarg_steers_the_run():
    @judge(name="business_metric")
    def business_metric(ctx):
        # A stand-in business rule: abort once the run passes 3 steps.
        if len(ctx) > 3:
            return stop("business rule tripped", meta={"steps": len(ctx)})
        return None

    session, interventions = asyncio.run(_run_with_judges([business_metric]))

    assert session.state.status.value == "aborted"
    corrections = [i for i in interventions if i["type"] == "correction"]
    assert corrections, "the custom judge should have been consulted"
    assert corrections[0]["judge"] == "business_metric"
    assert corrections[0]["flag"] == "stop"
    assert corrections[0]["meta"]["steps"] > 3


def test_judges_see_the_execution_input():
    seen = {}

    @judge(name="input_probe")
    def input_probe(ctx):
        seen.update(ctx.input)
        return stop("done probing")

    asyncio.run(_run_with_judges([input_probe]))
    assert seen.get("query") == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_integration.py -q -k "business_judge or execution_input"`
Expected: FAIL — `TypeError: SentinelRuntime.execute() got an unexpected keyword argument 'judges'`

- [ ] **Step 3: Add the `judges` parameter**

In `sentinal/runtime/runtime.py`, change the `execute` signature and the Governor block. Replace lines 123-126:

```python
    async def execute(
        self,
        session: ExecutionSession,
        judges=None,
    ) -> ExecutionSession:
```

and extend its docstring Args with:

```
            judges: Optional judges for the Governor. Overrides the judges of a
                pre-registered Governor, so callers no longer have to register
                one before calling execute().
```

Then replace the Governor creation block (lines 170-175):

```python
            if not governor:
                governor = Governor(
                    ActorRef("governor", "local"),
                    quality_threshold=self.quality_threshold,
                    judges=judges,
                )
                self.register(governor)
            elif judges is not None:
                logger.info(
                    "Overriding the registered Governor's judges with the "
                    "judges= passed to execute()")
                governor.set_judges(judges)
```

`Governor(judges=None)` falls through to the `HeuristicJudge()` default from Task 4, so the no-judges path is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_integration.py -q`
Expected: all pass (6 tests)

- [ ] **Step 5: Run the full suite**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add sentinal/runtime/runtime.py tests/test_trajectory/test_integration.py
git commit -m "feat: accept judges= on SentinelRuntime.execute"
```

---

### Task 6: Public exports and the example

**Files:**
- Modify: `sentinal/__init__.py`
- Modify: `examples/search_agent.py:59-71` (imports), `:436-518` (main)
- Test: `tests/test_judges/test_exports.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: the public API surface — `judge`, `redirect`, `avoid_tool`, `finalize`, `stop`, `JudgeContext`, `SEVERITY`, `StepBudgetJudge`, `RepeatToolJudge`, `OutputLoopJudge`, `LatencyBudgetJudge` all importable from `sentinal`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_judges/test_exports.py`:

```python
import sentinal


def test_public_judge_api_is_importable_from_the_package_root():
    names = [
        "judge", "redirect", "avoid_tool", "finalize", "stop",
        "JudgeContext", "SEVERITY",
        "StepBudgetJudge", "RepeatToolJudge", "OutputLoopJudge",
        "LatencyBudgetJudge", "HeuristicJudge",
    ]
    for n in names:
        assert hasattr(sentinal, n), f"sentinal.{n} is missing"
        assert n in sentinal.__all__, f"{n} missing from sentinal.__all__"


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_judges/test_exports.py -q`
Expected: FAIL — `AssertionError: sentinal.judge is missing`

- [ ] **Step 3: Update `sentinal/__init__.py`**

Replace the trajectory/judges import block (lines 14-20) with:

```python
from sentinal.core.trajectory import (
    SEVERITY,
    Correction,
    CorrectionFlag,
    JudgeContext,
    TrajectoryJudge,
    TrajectoryStep,
)
from sentinal.judges import (
    HeuristicJudge,
    LatencyBudgetJudge,
    OutputLoopJudge,
    RepeatToolJudge,
    StepBudgetJudge,
    avoid_tool,
    finalize,
    judge,
    redirect,
    stop,
)
```

and replace the tail of `__all__` (from `"CorrectionFlag"` onward) with:

```python
    "CorrectionFlag",
    "Correction",
    "TrajectoryStep",
    "TrajectoryJudge",
    "JudgeContext",
    "SEVERITY",
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
```

- [ ] **Step 4: Run the test**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_judges/test_exports.py -q`
Expected: 2 passed

- [ ] **Step 5: Update the example to show composition**

In `examples/search_agent.py`, add to the `from sentinal import (...)` block (line 59):

```python
    LatencyBudgetJudge,
    OutputLoopJudge,
    RepeatToolJudge,
    StepBudgetJudge,
    finalize,
    judge,
```

Add a user-defined business judge just above `# ─── Main ───` (after `DemoSteeringJudge`):

```python
# ─── A user-defined business judge ───────────────────────────────────────────
# Judges are plain functions. This one is the shape a business metric takes:
# read what you need off the context, return a correction or None.
@judge(name="answer_length_cap")
def answer_length_cap(ctx):
    """Wrap up once the agent has produced a long enough answer to be useful,
    so we don't pay for more search rounds than the question warrants."""
    chars = sum(len(s.summary) for s in ctx if s.kind == "agent")
    if chars > 2000:
        return finalize("answer is long enough; wrap up",
                        meta={"agent_chars": chars, "cap": 2000})
    return None
```

Replace the judge-selection block in `main()` (lines 447-454) with:

```python
    if demo_steering:
        judges = [DemoSteeringJudge()]
        executable = SearchAgentExecutable()
    elif tool_choice:
        tool_judge, executable, _ = make_tool_choice_run(query)
        judges = [tool_judge]
    else:
        # The curated built-ins composed with a user-defined business judge.
        judges = [
            StepBudgetJudge(max_steps=20),
            RepeatToolJudge(threshold=3),
            OutputLoopJudge(window=3),
            LatencyBudgetJudge(max_seconds=120),
            answer_length_cap,
        ]
        executable = SearchAgentExecutable()
```

Update the banner print (lines 461-467) to name all the judges:

```python
    if demo_steering:
        judge_desc = " (forces REDIRECT then STOP)"
    elif tool_choice:
        judge_desc = " (steers wrong tool → right tool)"
    else:
        judge_desc = " (fire only if the agent misbehaves)"
    names = ", ".join(getattr(j, "name", type(j).__name__) for j in judges)
    print(f"  Judges: {names}{judge_desc}")
```

Replace the Governor registration and `execute` call (lines 472-473 and 483/488) so judges flow through the runtime instead of a hand-registered Governor:

```python
    runtime = SentinelRuntime()
    session = ExecutionSession(
        executable=executable,
        context=ExecutionContext(input={"query": query}),
    )
```

and pass `judges=judges` to both `runtime.execute(session)` call sites:

```python
        result, _ = await asyncio.gather(
            runtime.execute(session, judges=judges),
            dash.run(session),
        )
    else:
        result = await runtime.execute(session, judges=judges)
```

The `Governor` and `ActorRef` imports become unused in the example — remove those two import lines (`from sentinal.core.actor_ref import ActorRef` and `from sentinal.actors.governor import Governor`).

Finally, show which judge spoke in the corrections summary (lines 513-515):

```python
    for c in corrections:
        note = f" — {c['note']}" if c.get("note") else ""
        suppressed = "" if c.get("pushed", True) else "  (suppressed by a more severe correction)"
        print(f"    • [{c.get('judge', '?')}] {c['flag']}{note}{suppressed}")
```

- [ ] **Step 6: Verify the example still imports and its judges wire up**

The example needs network + credentials to run fully, so verify statically:

Run: `PYTHON_GIL=0 .ft-env/bin/python -c "import ast,sys; ast.parse(open('examples/search_agent.py').read()); print('parses')"`
Expected: `parses`

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_tool_choice_steering.py -q`
Expected: passes (this test imports `make_tool_choice_run` from the example, so it catches import breakage).

- [ ] **Step 7: Run the whole suite**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/ -q`
Expected: all pass, no skips beyond the pre-existing 1.

- [ ] **Step 8: Commit**

```bash
git add sentinal/__init__.py examples/search_agent.py tests/test_judges/test_exports.py
git commit -m "feat: export judge API and compose built-ins in the search agent example"
```

---

## Verification

After all six tasks:

- [ ] `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/ -q` — all pass
- [ ] Baseline preserved: the 35 pre-existing `tests/test_trajectory/` tests still pass, with the two documented `HeuristicJudge` CONTINUE→None edits in Task 3 Step 5 the only intentional test change.
- [ ] `PYTHON_GIL=0 .ft-env/bin/python -c "import sentinal; print(len(sentinal.__all__))"` — imports cleanly
- [ ] A user can write a judge with only root imports (covered by `test_a_judge_can_be_written_using_only_root_imports`)
- [ ] `examples/search_agent.py:251-262` — the actor-side correction ladder is **unchanged**. If a task modified it, that task overstepped.

## Out of scope

Explicitly not part of this plan, per the spec:

- Extracting the duplicated actor-side correction ladder into a reusable helper.
- A built-in cost/spend judge (no provider-agnostic usage source).
- Promoting `ToolChoiceJudge` out of `examples/` (it puts a network call inside judging).
- Any new `CorrectionFlag` member.
