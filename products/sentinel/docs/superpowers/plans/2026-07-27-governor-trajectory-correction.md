# Governor Trajectory Judgment & Correction Flags — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Governor judge a streaming trajectory mid-flight (pluggable judge, heuristic default) and push typed correction flags the executable acts on between steps.

**Architecture:** New core types (`CorrectionFlag`, `TrajectoryStep`, `Correction`, `TrajectoryJudge` protocol) + a `HeuristicJudge`. `ExecutionState` gains two lock-guarded channels — trajectory (executable→Governor) and corrections (Governor→executable) — reusing its existing `self._lock`. The Governor judges new steps each poll and pushes non-CONTINUE corrections, emitting each to the live dashboard. The executable receives the live session via `ExecutionContext.session` and drains corrections between `astream` chunks; STOP raises `ExecutionAborted` → new `ExecutionStatus.ABORTED`.

**Tech Stack:** Python 3.13+, stdlib (`enum`, `dataclasses`, `threading`, `typing.Protocol`), `pytest`/`pytest-asyncio`. LangGraph only in the example (fakes in tests).

## Global Constraints

- Python `>=3.13`. Core `dependencies = []` stays empty — this feature is stdlib-only.
- Reuse `ExecutionState`'s existing `self._lock`; do NOT introduce a second lock or a new locking convention.
- Backward compatibility: non-streaming executables never call `append_step`, so `get_trajectory()` stays empty and no corrections are pushed. Existing `instructions` (RETRY/IMPROVE_QUALITY) and all existing tests must remain unchanged and green.
- A judge exception must be caught in `govern_session`, logged, and treated as `CONTINUE` — a broken judge must never crash the run.
- `CorrectionFlag.CONTINUE` is never pushed onto the corrections channel.
- STOP maps to a NEW distinct `ExecutionStatus.ABORTED` (not CANCELLED). `is_terminal()` must include ABORTED.
- `ExecutionContext` is `frozen=True, slots=True`; attach the session via `dataclasses.replace(context, session=session)`, never by mutation. The `session` field is typed `Any = None` to avoid a circular import with `session.py`.
- `HeuristicJudge` thresholds are constructor params with defaults (`repeat_threshold=3`, `step_budget=20`, `loop_window=3`) — no magic numbers inline.
- Tests use `.ft-env/bin/python -m pytest`. A pre-existing torch "Failed to initialize NumPy" UserWarning is environmental — ignore it. Two pre-existing broken files are excluded from suite runs: `test_openai.py` and `tests/test_integration.py`.

---

### Task 1: Core trajectory types

**Files:**
- Create: `sentinal/core/trajectory.py`
- Test: `tests/test_trajectory/test_types.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `class CorrectionFlag(str, Enum)`: `CONTINUE="continue"`, `REDIRECT="redirect"`, `AVOID_TOOL="avoid_tool"`, `FORCE_FINALIZE="force_finalize"`, `STOP="stop"`.
  - `@dataclass(frozen=True) TrajectoryStep`: `index: int`, `kind: str`, `summary: str`, `raw: dict`, `ts: float = 0.0`.
  - `@dataclass(frozen=True) Correction`: `flag: CorrectionFlag`, `note: str | None = None`, `step_index: int = -1`, `ts: float = 0.0`.
  - `class TrajectoryJudge(Protocol)`: `def judge(self, trajectory: list[TrajectoryStep]) -> Correction: ...`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory/__init__.py  -> create empty file
# tests/test_trajectory/test_types.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinal.core.trajectory'`

- [ ] **Step 3: Write minimal implementation**

```python
# sentinal/core/trajectory.py
"""Trajectory steps and Governor correction flags for mid-flight steering."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class CorrectionFlag(str, Enum):
    CONTINUE = "continue"            # all good, no-op (never pushed)
    REDIRECT = "redirect"            # steer: use note as guidance
    AVOID_TOOL = "avoid_tool"        # note = tool name to stop calling
    FORCE_FINALIZE = "force_finalize"  # wrap up now
    STOP = "stop"                    # abort the run


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


@runtime_checkable
class TrajectoryJudge(Protocol):
    def judge(self, trajectory: list[TrajectoryStep]) -> Correction: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_types.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/core/trajectory.py tests/test_trajectory/
git commit -m "feat: add trajectory step + correction flag core types"
```

---

### Task 2: HeuristicJudge

**Files:**
- Create: `sentinal/actors/judges.py`
- Test: `tests/test_trajectory/test_heuristic_judge.py`

**Interfaces:**
- Consumes: `CorrectionFlag`, `TrajectoryStep`, `Correction` (Task 1).
- Produces:
  - `class HeuristicJudge` with `__init__(self, *, repeat_threshold: int = 3, step_budget: int = 20, loop_window: int = 3)` and `judge(self, trajectory: list[TrajectoryStep]) -> Correction`.
  - Rule order (first match wins): (1) trajectory length `> step_budget` → `FORCE_FINALIZE`; (2) last `repeat_threshold` steps are all `kind=="tool"` with identical `summary` → `AVOID_TOOL` (note=that summary); (3) last `loop_window` `kind=="agent"` steps have identical `summary` → `REDIRECT` (note="repeated output; change approach"); (4) else `CONTINUE`. Empty trajectory → `CONTINUE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory/test_heuristic_judge.py
from sentinal.core.trajectory import CorrectionFlag, TrajectoryStep
from sentinal.actors.judges import HeuristicJudge


def _step(i, kind, summary):
    return TrajectoryStep(index=i, kind=kind, summary=summary, raw={})


def test_empty_trajectory_continues():
    assert HeuristicJudge().judge([]).flag is CorrectionFlag.CONTINUE


def test_healthy_trajectory_continues():
    traj = [_step(0, "agent", "think"), _step(1, "tool", "web_search"),
            _step(2, "agent", "answer")]
    assert HeuristicJudge().judge(traj).flag is CorrectionFlag.CONTINUE


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_heuristic_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinal.actors.judges'`

- [ ] **Step 3: Write minimal implementation**

```python
# sentinal/actors/judges.py
"""Default deterministic trajectory judge (no LLM)."""
from __future__ import annotations

from sentinal.core.trajectory import Correction, CorrectionFlag, TrajectoryStep


class HeuristicJudge:
    """Deterministic trajectory judge. Same trajectory -> same Correction."""

    def __init__(self, *, repeat_threshold: int = 3, step_budget: int = 20,
                 loop_window: int = 3) -> None:
        self._repeat_threshold = repeat_threshold
        self._step_budget = step_budget
        self._loop_window = loop_window

    def judge(self, trajectory: list[TrajectoryStep]) -> Correction:
        n = len(trajectory)
        if n == 0:
            return Correction(flag=CorrectionFlag.CONTINUE)

        # (1) step budget
        if n > self._step_budget:
            return Correction(flag=CorrectionFlag.FORCE_FINALIZE,
                              note="step budget exceeded", step_index=n - 1)

        # (2) repeated identical tool calls
        if n >= self._repeat_threshold:
            tail = trajectory[-self._repeat_threshold:]
            if all(s.kind == "tool" and s.summary == tail[0].summary for s in tail):
                return Correction(flag=CorrectionFlag.AVOID_TOOL,
                                  note=tail[0].summary, step_index=n - 1)

        # (3) agent output loop
        if n >= self._loop_window:
            tail = trajectory[-self._loop_window:]
            if all(s.kind == "agent" and s.summary == tail[0].summary for s in tail):
                return Correction(flag=CorrectionFlag.REDIRECT,
                                  note="repeated output; change approach",
                                  step_index=n - 1)

        return Correction(flag=CorrectionFlag.CONTINUE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_heuristic_judge.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/actors/judges.py tests/test_trajectory/test_heuristic_judge.py
git commit -m "feat: add HeuristicJudge default trajectory judge"
```

---

### Task 3: Session trajectory & correction channels

**Files:**
- Modify: `sentinal/core/execution/state.py`
- Modify: `sentinal/core/execution/session.py`
- Test: `tests/test_trajectory/test_channels.py`

**Interfaces:**
- Consumes: `TrajectoryStep`, `Correction` (Task 1).
- Produces (on both `ExecutionState` and, delegating, `ExecutionSession`):
  - `append_step(self, step: TrajectoryStep) -> None`
  - `get_trajectory(self) -> list[TrajectoryStep]` (snapshot copy)
  - `push_correction(self, c: Correction) -> None`
  - `drain_corrections(self) -> list[Correction]` (returns all pending, clears)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory/test_channels.py
import threading
from sentinal.core.execution.state import ExecutionState
from sentinal.core.trajectory import TrajectoryStep, Correction, CorrectionFlag


def _step(i):
    return TrajectoryStep(index=i, kind="agent", summary=f"s{i}", raw={})


def test_append_and_get_trajectory_snapshot():
    st = ExecutionState()
    st.append_step(_step(0))
    snap = st.get_trajectory()
    st.append_step(_step(1))
    assert len(snap) == 1          # snapshot is isolated
    assert len(st.get_trajectory()) == 2


def test_push_and_drain_corrections_once():
    st = ExecutionState()
    st.push_correction(Correction(flag=CorrectionFlag.REDIRECT, note="x"))
    drained = st.drain_corrections()
    assert len(drained) == 1
    assert drained[0].note == "x"
    assert st.drain_corrections() == []   # cleared after drain


def test_channels_thread_safe():
    st = ExecutionState()

    def writer():
        for i in range(100):
            st.append_step(_step(i))
            st.push_correction(Correction(flag=CorrectionFlag.CONTINUE))

    ts = [threading.Thread(target=writer) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(st.get_trajectory()) == 500
    assert len(st.drain_corrections()) == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_channels.py -v`
Expected: FAIL with `AttributeError: 'ExecutionState' object has no attribute 'append_step'`

- [ ] **Step 3: Write minimal implementation**

In `sentinal/core/execution/state.py`, add imports at top:

```python
from sentinal.core.trajectory import TrajectoryStep, Correction
```

In `ExecutionState.__init__`, after `self._instructions: List[str] = []`:

```python
        # Trajectory - executable writes, Governor reads
        self._trajectory: List[TrajectoryStep] = []
        # Corrections - Governor writes, executable drains
        self._corrections: List[Correction] = []
```

Add these methods (after the instructions section, reusing `self._lock`):

```python
    # ==========================================
    # Trajectory (executable writes, Governor reads)
    # ==========================================

    def append_step(self, step: TrajectoryStep) -> None:
        with self._lock:
            self._trajectory.append(step)

    def get_trajectory(self) -> List[TrajectoryStep]:
        with self._lock:
            return self._trajectory.copy()

    # ==========================================
    # Corrections (Governor writes, executable drains)
    # ==========================================

    def push_correction(self, correction: Correction) -> None:
        with self._lock:
            self._corrections.append(correction)

    def drain_corrections(self) -> List[Correction]:
        with self._lock:
            drained = self._corrections.copy()
            self._corrections.clear()
            return drained
```

In `sentinal/core/execution/session.py`, add delegating convenience methods after `get_instructions`:

```python
    def append_step(self, step) -> None:
        """Append a trajectory step (executable → Governor)."""
        self.state.append_step(step)

    def get_trajectory(self) -> list:
        """Get trajectory snapshot (Governor reads)."""
        return self.state.get_trajectory()

    def push_correction(self, correction) -> None:
        """Push a correction (Governor → executable)."""
        self.state.push_correction(correction)

    def drain_corrections(self) -> list:
        """Drain pending corrections (executable reads & clears)."""
        return self.state.drain_corrections()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_channels.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/core/execution/state.py sentinal/core/execution/session.py tests/test_trajectory/test_channels.py
git commit -m "feat: add trajectory + correction channels to session state"
```

---

### Task 4: ABORTED status + ExecutionAborted exception

**Files:**
- Modify: `sentinal/core/execution/snapshot.py`
- Modify: `sentinal/core/execution/state.py`
- Modify: `sentinal/exceptions/runtime.py`
- Test: `tests/test_trajectory/test_aborted_state.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ExecutionStatus.ABORTED = "aborted"` enum member.
  - `ExecutionState.is_terminal()` returns True for ABORTED (in addition to COMPLETED/FAILED/CANCELLED).
  - `class ExecutionAborted(SentinelRuntimeError)` in exceptions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory/test_aborted_state.py
from dataclasses import replace
from datetime import UTC, datetime
from sentinal.core.execution.state import ExecutionState
from sentinal.core.execution.snapshot import ExecutionStatus
from sentinal.exceptions.runtime import ExecutionAborted, SentinelRuntimeError


def test_aborted_status_exists():
    assert ExecutionStatus.ABORTED.value == "aborted"


def test_aborted_is_terminal():
    st = ExecutionState()
    st._current = replace(st._current, status=ExecutionStatus.ABORTED,
                          updated_at=datetime.now(UTC))
    assert st.is_terminal() is True


def test_execution_aborted_is_runtime_error():
    assert issubclass(ExecutionAborted, SentinelRuntimeError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_aborted_state.py -v`
Expected: FAIL with `AttributeError: ABORTED` (enum member missing)

- [ ] **Step 3: Write minimal implementation**

In `sentinal/core/execution/snapshot.py`, add to `ExecutionStatus`:

```python
    ABORTED = "aborted"
```

In `sentinal/core/execution/state.py`, `is_terminal()`:

```python
    def is_terminal(self) -> bool:
        """Check if execution is in terminal state."""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.ABORTED,
        ]
```

In `sentinal/exceptions/runtime.py`, add:

```python
class ExecutionAborted(SentinelRuntimeError):
    """Raised when the Governor stops a run mid-flight (STOP correction)."""
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_aborted_state.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/core/execution/snapshot.py sentinal/core/execution/state.py sentinal/exceptions/runtime.py tests/test_trajectory/test_aborted_state.py
git commit -m "feat: add ABORTED status and ExecutionAborted exception"
```

---

### Task 5: ExecutionContext carries the live session

**Files:**
- Modify: `sentinal/core/execution/context.py`
- Test: `tests/test_trajectory/test_context_session.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ExecutionContext.session: Any = None` field; `replace(context, session=...)` works.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory/test_context_session.py
from dataclasses import replace
from sentinal.core.execution.context import ExecutionContext


def test_context_session_defaults_none():
    ctx = ExecutionContext(input={"q": 1})
    assert ctx.session is None


def test_context_session_via_replace():
    ctx = ExecutionContext(input={"q": 1})
    sentinel = object()
    ctx2 = replace(ctx, session=sentinel)
    assert ctx2.session is sentinel
    assert ctx.session is None          # original unchanged (frozen)
    assert ctx2.input == {"q": 1}       # other fields preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_context_session.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'session'` (or AttributeError on `.session`)

- [ ] **Step 3: Write minimal implementation**

In `sentinal/core/execution/context.py`, add field to `ExecutionContext` (after `capabilities`):

```python
    session: Any = None
    """Live ExecutionSession when running under the runtime, else None.
    Typed Any to avoid a circular import with session.py."""
```

(`Any` is already imported.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_context_session.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/core/execution/context.py tests/test_trajectory/test_context_session.py
git commit -m "feat: add optional live session field to ExecutionContext"
```

---

### Task 6: Governor judges trajectory & pushes corrections

**Files:**
- Modify: `sentinal/actors/governor.py`
- Test: `tests/test_trajectory/test_governor_corrections.py`

**Interfaces:**
- Consumes: `HeuristicJudge` (Task 2), `TrajectoryJudge`/`Correction`/`CorrectionFlag` (Task 1), session channels (Task 3), `activity_bus` (existing).
- Produces:
  - `Governor.__init__(self, *args, quality_threshold: float = 0.7, judge: TrajectoryJudge | None = None, **kwargs)` — defaults `self._judge = judge or HeuristicJudge()`; adds `self._last_judged_len = 0`.
  - In `govern_session`, before the existing status checks each poll: judge new trajectory and push non-CONTINUE corrections, emitting to the bus. Judge exceptions caught → treated as CONTINUE.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory/test_governor_corrections.py
import asyncio
from sentinal.core.actor_ref import ActorRef
from sentinal.actors.governor import Governor
from sentinal.core.trajectory import (
    Correction, CorrectionFlag, TrajectoryStep,
)
from sentinal.telemetry.dashboard import activity_bus


class _FakeJudge:
    def __init__(self, correction):
        self._c = correction
        self.calls = 0

    def judge(self, trajectory):
        self.calls += 1
        return self._c


class _Session:
    """Fake session exposing the channels the Governor uses."""
    def __init__(self, steps, polls):
        self.session_id = "sess-gov"
        self._steps = steps
        self._pushed = []
        self._polls = polls

    def get_trajectory(self):
        return list(self._steps)

    def push_correction(self, c):
        self._pushed.append(c)

    def get_all_memory(self):
        return {}

    def is_terminal(self):
        self._polls -= 1
        return self._polls <= 0


def _run_governor(session, judge):
    gov = Governor(ActorRef("governor", "local"), judge=judge)
    asyncio.run(gov.govern_session(session, poll_interval=0.001))
    return gov


def test_governor_pushes_non_continue_correction():
    steps = [TrajectoryStep(index=0, kind="tool", summary="web_search", raw={})]
    judge = _FakeJudge(Correction(flag=CorrectionFlag.REDIRECT, note="try X"))
    session = _Session(steps, polls=3)
    _run_governor(session, judge)
    assert any(c.flag is CorrectionFlag.REDIRECT for c in session._pushed)


def test_governor_does_not_push_continue():
    steps = [TrajectoryStep(index=0, kind="agent", summary="ok", raw={})]
    judge = _FakeJudge(Correction(flag=CorrectionFlag.CONTINUE))
    session = _Session(steps, polls=3)
    _run_governor(session, judge)
    assert session._pushed == []


def test_governor_does_not_re_push_on_unchanged_trajectory():
    steps = [TrajectoryStep(index=0, kind="tool", summary="web_search", raw={})]
    judge = _FakeJudge(Correction(flag=CorrectionFlag.REDIRECT))
    session = _Session(steps, polls=5)
    _run_governor(session, judge)
    # Only judged once because trajectory length never grew
    assert len(session._pushed) == 1


def test_governor_survives_judge_exception():
    class _Boom:
        def judge(self, trajectory):
            raise RuntimeError("judge broke")
    steps = [TrajectoryStep(index=0, kind="tool", summary="x", raw={})]
    session = _Session(steps, polls=3)
    _run_governor(session, _Boom())   # must not raise
    assert session._pushed == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_governor_corrections.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'judge'`

- [ ] **Step 3: Write minimal implementation**

In `sentinal/actors/governor.py`, add imports at top:

```python
from sentinal.actors.judges import HeuristicJudge
from sentinal.core.trajectory import CorrectionFlag, TrajectoryJudge
from sentinal.telemetry.dashboard import activity_bus
```

Change `__init__`:

```python
    def __init__(self, *args, quality_threshold: float = 0.7,
                 judge: TrajectoryJudge | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_threshold = quality_threshold
        self._interventions = []
        self._judge = judge or HeuristicJudge()
        self._last_judged_len = 0
```

In `govern_session`, inside the `while not session.is_terminal():` loop, BEFORE the existing `memory = session.state.get_all_memory()` status logic, add:

```python
            # Judge the trajectory as it grows and push corrections
            trajectory = session.get_trajectory()
            if len(trajectory) > self._last_judged_len:
                self._last_judged_len = len(trajectory)
                try:
                    correction = self._judge.judge(trajectory)
                except Exception as ex:   # a broken judge must not crash the run
                    logger.warning(f"Governor judge failed: {ex}")
                    correction = None
                if correction is not None and correction.flag != CorrectionFlag.CONTINUE:
                    session.push_correction(correction)
                    activity_bus.emit(
                        "Governor", "correction",
                        status="governing",
                        detail=f"{correction.flag.value}:{correction.note or ''}",
                        session_id=str(session.session_id),
                    )
                    self._interventions.append({
                        "type": "correction",
                        "flag": correction.flag.value,
                        "note": correction.note,
                    })
```

Note: the fake `_Session` in the test does not have `session.state`; the existing status logic calls `session.state.get_all_memory()`. To keep the test session minimal, the existing code already calls `memory = session.state.get_all_memory()`. Update the test's `_Session` is not allowed here — instead, guard the existing memory read defensively is NOT needed because the test provides `get_all_memory` directly on the session. VERIFY: the current governor uses `session.state.get_all_memory()`. Keep that line as-is; the fake provides `get_all_memory` on the session itself, so add a `state` shim in the test. (Handled in the test: see below fix.)

IMPORTANT: adjust the test `_Session` to expose `state` — replace its `get_all_memory` with a nested `state` object:

```python
    # in _Session.__init__:  self.state = self
    # keep get_all_memory(self) returning {}
```

Add `self.state = self` in `_Session.__init__` so `session.state.get_all_memory()` resolves.

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_governor_corrections.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/actors/governor.py tests/test_trajectory/test_governor_corrections.py
git commit -m "feat: governor judges trajectory and pushes correction flags"
```

---

### Task 7: Executor attaches session & handles ExecutionAborted

**Files:**
- Modify: `sentinal/actors/executor.py`
- Modify: `sentinal/telemetry/dashboard.py`
- Test: `tests/test_trajectory/test_executor_abort.py`

**Interfaces:**
- Consumes: `ExecutionAborted` (Task 4), `ExecutionStatus.ABORTED` (Task 4), `replace` on context (Task 5).
- Produces:
  - Executor rebuilds context: `context = replace(session.context, session=session)` and calls `session.executable.execute(context)` with it.
  - New `except ExecutionAborted as ex:` clause BEFORE the generic `except Exception`, setting `actor_status="aborted"`, transitioning snapshot to `ExecutionStatus.ABORTED`, publishing `ExecutionFailed`, and (via memory) surfacing the abort. Does not re-raise.
  - `dashboard.py`: `_STATUS_STYLE["aborted"] = "red"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory/test_executor_abort.py
import asyncio
from sentinal.core.actor_ref import ActorRef
from sentinal.actors.executor import Executor
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.execution.snapshot import ExecutionStatus
from sentinal.exceptions.runtime import ExecutionAborted


class _AbortingExecutable(Executable):
    async def execute(self, context):
        # prove the live session was attached to the context
        assert context.session is not None
        raise ExecutionAborted("stopped by governor")


class _OkExecutable(Executable):
    async def execute(self, context):
        assert context.session is not None
        return {"ok": True}


def test_executor_marks_aborted_on_execution_aborted():
    session = ExecutionSession(executable=_AbortingExecutable(),
                               context=ExecutionContext(input={}))
    ex = Executor(ActorRef("executor", "local"))
    asyncio.run(ex.execute_session(session))
    assert session.state.status is ExecutionStatus.ABORTED
    assert session.get_memory("actor_status") == "aborted"


def test_executor_attaches_session_to_context_on_success():
    session = ExecutionSession(executable=_OkExecutable(),
                               context=ExecutionContext(input={}))
    ex = Executor(ActorRef("executor", "local"))
    asyncio.run(ex.execute_session(session))
    assert session.state.status is ExecutionStatus.COMPLETED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_executor_abort.py -v`
Expected: FAIL — `test_executor_marks_aborted...` fails because ExecutionAborted is caught by the generic handler and status becomes FAILED (and/or `context.session` assertion fails: session not attached).

- [ ] **Step 3: Write minimal implementation**

In `sentinal/actors/executor.py`, add imports at top:

```python
from dataclasses import replace
from sentinal.exceptions.runtime import ExecutionAborted
```

In `execute_session`, replace the executable call so the live session is attached. Find:

```python
            # Execute the workload
            result = await session.executable.execute(session.context)
```

Replace with:

```python
            # Attach the live session so streaming executables can steer
            context = replace(session.context, session=session)
            # Execute the workload
            result = await session.executable.execute(context)
```

Add a dedicated abort handler BEFORE the existing `except asyncio.CancelledError:` / `except Exception as ex:` blocks (order matters — must precede generic Exception):

```python
        except ExecutionAborted as ex:
            logger.warning(f"Executor aborted by governor: {ex}")
            session.update_memory("actor_status", "aborted")
            session.update_memory("abort_reason", str(ex))

            new_snapshot = replace(
                session.state._current,
                status=ExecutionStatus.ABORTED,
                updated_at=datetime.now(UTC),
            )
            session.state._current = new_snapshot
            session.state._history.append(new_snapshot)

            await self.publish(
                ExecutionFailed(
                    execution_id=session.execution_id,
                    sender=self.ref,
                    data=ex,
                )
            )
```

(`ExecutionStatus`, `datetime`, `UTC`, `ExecutionFailed`, `replace` are already imported in executor.py except `replace`/`ExecutionAborted` which you added. Verify `ExecutionStatus` import exists — it is imported inside `execute_session` in the current code via `from sentinal.core.execution.snapshot import ExecutionStatus`; ensure the abort block is within scope of that import or add a top-level import.)

In `sentinal/telemetry/dashboard.py`, add to `_STATUS_STYLE`:

```python
    "aborted": "red",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_executor_abort.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/actors/executor.py sentinal/telemetry/dashboard.py tests/test_trajectory/test_executor_abort.py
git commit -m "feat: executor attaches session and handles ExecutionAborted -> ABORTED"
```

---

### Task 8: End-to-end steering integration test

**Files:**
- Test: `tests/test_trajectory/test_integration.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7 + `SentinelRuntime` (existing).
- Produces: a test proving a correction flows executable→Governor→executable across threads and changes the outcome.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_trajectory/test_integration.py
import asyncio
from sentinal.runtime.runtime import SentinelRuntime
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.execution.snapshot import ExecutionStatus
from sentinal.core.trajectory import TrajectoryStep
from sentinal.exceptions.runtime import ExecutionAborted
from sentinal.actors.judges import HeuristicJudge


class _StreamingExecutable(Executable):
    """Emits many identical tool steps; drains corrections; aborts on STOP,
    finalizes on FORCE_FINALIZE."""
    async def execute(self, context):
        session = context.session
        assert session is not None
        i = 0
        while i < 50:
            session.append_step(
                TrajectoryStep(index=i, kind="tool", summary="web_search", raw={})
            )
            for c in session.drain_corrections():
                if c.flag.value == "stop":
                    raise ExecutionAborted(c.note or "stop")
                if c.flag.value in ("force_finalize", "avoid_tool", "redirect"):
                    return {"finalized_at": i, "flag": c.flag.value}
            await asyncio.sleep(0.005)
            i += 1
        return {"finalized_at": i}


async def _run():
    runtime = SentinelRuntime()
    session = ExecutionSession(
        executable=_StreamingExecutable(),
        context=ExecutionContext(input={}),
    )
    await runtime.execute(session)
    await runtime.shutdown()
    return session


def test_correction_flows_end_to_end_and_finalizes_early():
    session = asyncio.run(_run())
    memory = session.state.get_all_memory()
    result = memory.get("result", {})
    # HeuristicJudge fires AVOID_TOOL after 3 identical tool calls, so the
    # executable returns well before its 50-step cap.
    assert result.get("finalized_at", 999) < 10
    assert result.get("flag") in ("avoid_tool", "redirect", "force_finalize")
```

- [ ] **Step 2: Run test to verify behavior**

Run: `.ft-env/bin/python -m pytest tests/test_trajectory/test_integration.py -v`
Expected: PASS — the run finalizes early (correction delivered across threads). If it fails because no correction arrives, that indicates a wiring regression in Tasks 6/7.

- [ ] **Step 3: Commit**

```bash
git add tests/test_trajectory/test_integration.py
git commit -m "test: end-to-end governor trajectory correction across threads"
```

---

### Task 9: Exports + streaming example + suite green

**Files:**
- Modify: `sentinal/__init__.py`
- Modify: `examples/search_agent.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all public types.
- Produces: `sentinal` exports `CorrectionFlag`, `Correction`, `TrajectoryStep`, `TrajectoryJudge`, `HeuristicJudge`; the example streams and steers; README documents the feature.

- [ ] **Step 1: Add exports**

In `sentinal/__init__.py`, add:

```python
from sentinal.core.trajectory import (
    CorrectionFlag,
    Correction,
    TrajectoryStep,
    TrajectoryJudge,
)
from sentinal.actors.judges import HeuristicJudge
```

and add `"CorrectionFlag"`, `"Correction"`, `"TrajectoryStep"`, `"TrajectoryJudge"`, `"HeuristicJudge"` to `__all__`.

- [ ] **Step 2: Verify exports import**

Run: `.ft-env/bin/python -c "from sentinal import CorrectionFlag, Correction, TrajectoryStep, TrajectoryJudge, HeuristicJudge; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Make the example executable stream & steer**

In `examples/search_agent.py`, change `SearchAgentExecutable.execute` to stream via `self._agent.astream(..., stream_mode="updates")`, appending a `TrajectoryStep` per chunk (`kind` inferred: chunk key `"tools"` → `"tool"` with the tool name as summary, else `"agent"` with a content preview) via `context.session.append_step(...)` when `context.session is not None`, and draining corrections between chunks:

```python
    @traced(name="agent.execute")
    @logged(capture_result=False)
    async def execute(self, context):
        query = context.input.get("query", "Hello")
        session = context.session
        inputs = {"messages": [HumanMessage(content=query)]}
        last = None
        i = 0
        async for chunk in self._agent.astream(inputs, stream_mode="updates"):
            last = chunk
            if session is not None:
                node = next(iter(chunk)) if chunk else "agent"
                kind = "tool" if node == "tools" else "agent"
                session.append_step(TrajectoryStep(
                    index=i, kind=kind, summary=str(node),
                    raw=chunk, ts=time.monotonic()))
                for c in session.drain_corrections():
                    if c.flag == CorrectionFlag.STOP:
                        raise ExecutionAborted(c.note or "stopped by governor")
                    elif c.flag in (CorrectionFlag.FORCE_FINALIZE,
                                    CorrectionFlag.REDIRECT,
                                    CorrectionFlag.AVOID_TOOL):
                        inputs = {"messages": [HumanMessage(
                            content=f"[governor:{c.flag.value}] {c.note or 'adjust course'}")]}
            i += 1
        # extract final answer from last chunk's messages
        final_text = ""
        if last:
            for v in last.values():
                msgs = v.get("messages") if isinstance(v, dict) else None
                if msgs:
                    final_text = getattr(msgs[-1], "content", "") or final_text
        return {"query": query, "response": final_text,
                "steps": i, "thread_id": threading.current_thread().native_id}
```

Add imports at the top of the example: `from sentinal import CorrectionFlag, TrajectoryStep` and `from sentinal.exceptions.runtime import ExecutionAborted`.

- [ ] **Step 4: Compile-check the example**

Run: `.ft-env/bin/python -m py_compile examples/search_agent.py && echo compiled`
Expected: `compiled`

- [ ] **Step 5: README note**

Append a "## Governor trajectory correction" section to `README.md` documenting: the Governor judges the streaming trajectory with a pluggable `TrajectoryJudge` (default `HeuristicJudge`), pushes typed `Correction` flags (REDIRECT / AVOID_TOOL / FORCE_FINALIZE / STOP), the executable acts on them between steps, STOP yields `ExecutionStatus.ABORTED`, and each correction shows in the live dashboard's Governor row. Ensure a single trailing newline.

- [ ] **Step 6: Full suite green**

Run: `.ft-env/bin/python -m pytest -q --ignore=test_openai.py --ignore=tests/test_integration.py`
Expected: all pass (existing 118 + new trajectory tests).

- [ ] **Step 7: Commit**

```bash
git add sentinal/__init__.py examples/search_agent.py README.md
git commit -m "feat: export trajectory types, stream+steer example, document feature"
```

---

## Self-Review

**Spec coverage:**
- Core types (CorrectionFlag/TrajectoryStep/Correction/TrajectoryJudge) → Task 1 ✓
- HeuristicJudge rules (repeat, budget, loop, continue) → Task 2 ✓
- Session channels (trajectory + corrections, thread-safe, reuse lock) → Task 3 ✓
- ABORTED status + is_terminal + ExecutionAborted → Task 4 ✓
- Session on ExecutionContext via replace (frozen) → Task 5 ✓
- Governor judge + push + emit + no-re-push + judge-exception-safe → Task 6 ✓
- Executor attach session + ExecutionAborted→ABORTED, dashboard color → Task 7 ✓
- End-to-end cross-thread integration → Task 8 ✓
- Exports + streaming example + README + suite green → Task 9 ✓
- Backward compat (non-streaming → empty trajectory → no corrections) → guaranteed by Task 3 semantics + Task 6 length check; existing tests re-run in Task 9 ✓

**Placeholder scan:** No TBD/TODO. Task 6 Step 3 contains a self-correcting note about the fake `_Session` needing a `state` shim; the required `self.state = self` line is stated explicitly — implementer must apply it. All code steps show full code.

**Type consistency:** `judge(trajectory) -> Correction` identical across Tasks 1/2/6. Channel method names (`append_step`/`get_trajectory`/`push_correction`/`drain_corrections`) identical across Tasks 3/6/7/8. `CorrectionFlag` members identical across Tasks 1/2/6/7/9. `ExecutionStatus.ABORTED` consistent Tasks 4/7. `context.session` consistent Tasks 5/7/9.

Note: Task 6's test fake exposes both `get_trajectory`/`push_correction` (governor's new path) and `state.get_all_memory` (governor's existing path); the plan calls this out explicitly so the implementer wires the `state = self` shim.
