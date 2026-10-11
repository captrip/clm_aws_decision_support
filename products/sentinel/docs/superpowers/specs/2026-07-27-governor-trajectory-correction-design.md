# Governor Trajectory Judgment & Correction Flags — Design

**Date:** 2026-07-27
**Status:** Approved (pending spec review)
**Author:** Sentinal team

## Problem

Today the Governor cannot steer a run in progress. Its `govern_session` loop
(`sentinal/actors/governor.py`) only reacts to terminal-ish states: it adds a
`"RETRY"` instruction when `actor_status == "failed"`, or computes a static
quality score on `"completed"` and adds `"IMPROVE_QUALITY"` / `"USE_BETTER_TOOLS"`.
Crucially, **nothing consumes those instructions mid-flight** — the Executor runs
`session.executable.execute(context)` as one opaque `ainvoke` to completion. The
Governor labels the outcome; it does not influence the trajectory.

We want the Governor to (1) judge the current trajectory as it unfolds, and
(2) pass typed correction flags that the executable acts on **between steps**.

## Goal

Mid-flight steering: as a streaming executable emits trajectory steps, the Governor
judges the trajectory-so-far with a **pluggable judge** (heuristic default) and pushes
**typed correction flags with optional notes** onto a thread-safe channel that the
executable drains and acts on before its next step.

## Non-Goals

- No LLM-based judge in v1. The `TrajectoryJudge` protocol is designed so an LLM judge
  drops in later, but only the deterministic `HeuristicJudge` ships now (YAGNI).
- No change to non-streaming executables. If an executable does not emit a trajectory,
  the Governor has nothing to judge and behavior is unchanged (backward-compatible).
- No new locking convention. `ExecutionState` already holds a `threading.Lock` and
  guards all memory/instruction access; the new channels reuse it.
- The existing `instructions` list (RETRY/IMPROVE_QUALITY) stays as-is for
  backward-compat; corrections are a separate, richer mid-flight channel.

## Architecture & data flow

```
Executable thread                     Governor thread
─────────────────                     ───────────────
langgraph astream                     govern_session loop (~10Hz poll)
  │ each step                              │
  ├─ append TrajectoryStep ──► session.trajectory ──► get_trajectory() (snapshot)
  │                            (lock-guarded)          │
  │                                                     ├─ TrajectoryJudge.judge(steps)
  ◄─ poll corrections ◄── session.corrections ◄─────────┤   → Correction(flag, note)
  │  between steps        (lock-guarded)                │  (push only if flag != CONTINUE)
  └─ act on Correction (STOP / REDIRECT / FORCE_FINALIZE / AVOID_TOOL / CONTINUE)
```

The Executor thread runs the executable; the Governor runs on its own CPU thread
(`govern_session`). The trajectory (executable → Governor) and corrections
(Governor → executable) cross threads, so both channels are lock-guarded on
`ExecutionState`, following the existing `update_memory` pattern.

## Core types

New module `sentinal/core/trajectory.py`:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Protocol

class CorrectionFlag(str, Enum):
    CONTINUE       = "continue"        # all good, no-op (never pushed)
    REDIRECT       = "redirect"        # steer: use note as guidance
    AVOID_TOOL     = "avoid_tool"      # note = tool name to stop calling
    FORCE_FINALIZE = "force_finalize"  # wrap up now, produce best answer
    STOP           = "stop"            # abort the run

@dataclass(frozen=True)
class TrajectoryStep:
    index: int
    kind: str          # "agent" | "tool" | "message"
    summary: str       # short human-readable (tool name / content preview)
    raw: dict          # the streamed chunk, for the judge
    ts: float

@dataclass(frozen=True)
class Correction:
    flag: CorrectionFlag
    note: str | None = None    # optional free-text (hint, tool name, reason)
    step_index: int = -1       # trajectory index that triggered it
    ts: float = 0.0

class TrajectoryJudge(Protocol):
    def judge(self, trajectory: list[TrajectoryStep]) -> Correction: ...
```

### HeuristicJudge (default)

Deterministic, no LLM, no threads — in `sentinal/actors/judges.py`:

- **Repeated tool:** same tool `kind == "tool"` with the same `summary` called
  `>= repeat_threshold` (default 3) consecutive times → `AVOID_TOOL` (note = tool name)
  the first time, escalating to `REDIRECT` if it persists.
- **Step budget:** `len(trajectory) > step_budget` (default 20) → `FORCE_FINALIZE`.
- **Loop:** the last `loop_window` (default 3) agent-step summaries are all identical →
  `REDIRECT` (note = "detected repeated output; change approach").
- **Otherwise:** `CONTINUE`.

Thresholds are constructor params with the defaults above (no magic numbers inline).
The judge is pure: same trajectory → same `Correction`.

## Session channels & thread safety

Add to `ExecutionState` (reusing its existing `self._lock`), and delegate from
`ExecutionSession` (matching the `update_memory` convention):

```python
# ExecutionState
def append_step(self, step: TrajectoryStep) -> None       # lock; self._trajectory.append
def get_trajectory(self) -> list[TrajectoryStep]          # lock; return copy
def push_correction(self, c: Correction) -> None          # lock; self._corrections.append
def drain_corrections(self) -> list[Correction]           # lock; return + clear

# ExecutionSession convenience delegates (same names)
```

- `self._trajectory: list[TrajectoryStep] = []` — append-only; `get_trajectory` returns
  a snapshot copy so the Governor never iterates a mutating list.
- `self._corrections: list[Correction] = []` — `drain_corrections` returns all pending
  and clears, so each correction is applied at most once.

## Governor behavior

`Governor.__init__` gains `judge: TrajectoryJudge | None = None`, defaulting to
`HeuristicJudge()`. Existing failure/quality logic is unchanged. In `govern_session`,
each poll while not terminal:

```python
steps = session.get_trajectory()
if len(steps) > self._last_judged_len:
    correction = self._judge.judge(steps)
    self._last_judged_len = len(steps)
    if correction.flag != CorrectionFlag.CONTINUE:
        session.push_correction(correction)
        activity_bus.emit("Governor", "correction",
                          status="governing",
                          detail=f"{correction.flag.value}:{correction.note or ''}",
                          session_id=str(session.session_id))
        self._interventions.append({"type": "correction",
                                    "flag": correction.flag.value,
                                    "note": correction.note})
```

The emit surfaces every correction in the live thread dashboard (built previously), so
steering is visible in the Governor row. To avoid re-pushing the same correction each
poll, the Governor tracks `_last_judged_len` and only re-judges when new steps arrive.

## Executable steering (LangGraph example)

The example executable becomes streaming. It drives `agent.astream(...,
stream_mode="updates")`, appends a `TrajectoryStep` per chunk, and drains corrections
between chunks:

```python
async for chunk in self._agent.astream(inputs, stream_mode="updates"):
    session.append_step(TrajectoryStep(index=i, kind=..., summary=..., raw=chunk, ts=...))
    for c in session.drain_corrections():
        if c.flag == CorrectionFlag.STOP:
            raise ExecutionAborted(c.note or "stopped by governor")
        elif c.flag == CorrectionFlag.FORCE_FINALIZE:
            inputs = _append_message(inputs, "Finalize your answer now.")
            finalize_next = True
        elif c.flag == CorrectionFlag.REDIRECT:
            inputs = _append_message(inputs, c.note or "reconsider your approach")
        elif c.flag == CorrectionFlag.AVOID_TOOL:
            inputs = _append_message(inputs, f"Do not use the {c.note} tool again.")
    if finalize_next:
        break
final = _last_message(chunk)
```

`session` reaches the executable via `ExecutionContext` (the executable already receives
`context`; add the session or a steering handle to it — see Open decision below).

## Error handling & states

- **STOP:** executable raises `ExecutionAborted` (new exception in
  `sentinal/exceptions/runtime.py`). The Executor catches it **before** its generic
  `except Exception` (order matters — a dedicated `except ExecutionAborted` clause first),
  sets `actor_status = "aborted"`, and transitions the snapshot to a **new**
  `ExecutionStatus.ABORTED` (distinct from CANCELLED, which is user/system cancellation;
  ABORTED specifically means the Governor stopped the run). Emits an `aborted` event to
  the dashboard. `ExecutionState.is_terminal()` must include `ABORTED`; the dashboard's
  `_STATUS_STYLE` gets an `aborted` → red entry.
- **Late corrections:** any correction drained after the agent has already finalized is
  ignored (loop has exited). No effect, no error.
- **Non-streaming executable:** never calls `append_step`, so `get_trajectory()` stays
  empty, the Governor's judge sees nothing, and no corrections are pushed. Fully
  backward-compatible.
- **Judge exceptions:** `govern_session` wraps `self._judge.judge(...)` in try/except;
  a judge failure is logged and treated as `CONTINUE` — a broken judge must never crash
  the run.

## Steering handle (decided)

The executable obtains its steering handle via the **live session on
`ExecutionContext`**. `ExecutionContext` gains an optional `session` field (default
`None`), so the streaming executable calls `context.session.append_step(...)` and
`context.session.drain_corrections()`. This requires no change to the
`Executable.execute(context)` signature.

`ExecutionContext` is `frozen=True, slots=True`, so it cannot be mutated after
construction. The Executor therefore **rebuilds** the context at the top of
`execute_session`, before calling the executable:

```python
from dataclasses import replace
context = replace(session.context, session=session)   # attach live session
...
result = await session.executable.execute(context)
```

The new `session` field is typed `Any = None` (not the `ExecutionSession` type) to avoid
a circular import between `context.py` and `session.py`; it is documented as "the live
ExecutionSession when running under the runtime, else None". Non-streaming executables
ignore it; executables run outside a session see `None` and skip steering.

## Testing

- **HeuristicJudge** — one unit test per rule (repeated tool → AVOID_TOOL/REDIRECT,
  budget exceeded → FORCE_FINALIZE, loop → REDIRECT, healthy → CONTINUE) with
  hand-built `TrajectoryStep` lists. Pure, no LLM, no threads.
- **Session channels** — append/get snapshot isolation; push/drain-once semantics;
  concurrent multi-thread access safety (mirrors the dashboard bus concurrency test).
- **Governor** — inject a fake judge; feed a session whose `get_trajectory()` returns a
  scripted step stream; assert it pushes expected corrections, emits to the bus, and
  does not re-push on unchanged trajectory. Assert judge-exception → treated as CONTINUE.
- **Executable steering** — a fake steppable executable (no real LLM) driven by
  pre-queued corrections: assert STOP raises ExecutionAborted, FORCE_FINALIZE ends
  early, REDIRECT/AVOID_TOOL inject the note.
- **Integration** — one test through `runtime.execute` with a fake streaming executable
  + real Governor + HeuristicJudge, asserting a correction flows end-to-end across the
  Executor and Governor threads and changes the run's outcome.
- The real LangGraph example is wired but requires credentials for a live run; automated
  tests use fakes.

## Files touched

- **New:** `sentinal/core/trajectory.py` (CorrectionFlag, TrajectoryStep, Correction,
  TrajectoryJudge protocol)
- **New:** `sentinal/actors/judges.py` (HeuristicJudge)
- **New:** `tests/test_trajectory/` (judge, channels, governor, steering, integration)
- **Edit:** `sentinal/core/execution/state.py` — trajectory + correction channels
- **Edit:** `sentinal/core/execution/session.py` — delegate methods
- **Edit:** `sentinal/actors/governor.py` — judge + push corrections + emit
- **Edit:** `sentinal/core/execution/snapshot.py` — add `ExecutionStatus.ABORTED`
- **Edit:** `sentinal/core/execution/state.py` — `is_terminal()` includes ABORTED; add the
  trajectory/correction channels. (The Executor sets the ABORTED snapshot inline via
  `replace(...)` exactly as it already does for RUNNING/COMPLETED/FAILED, so no new event
  type is needed in `apply()`.)
- **Edit:** `sentinal/actors/executor.py` — rebuild context with
  `replace(session.context, session=session)` before dispatch; `except ExecutionAborted`
  (before generic Exception) → status "aborted" / `ExecutionStatus.ABORTED`, emit `aborted`
- **Edit:** `sentinal/core/execution/context.py` — add optional `session: Any = None` field
- **Edit:** `sentinal/telemetry/dashboard.py` — `_STATUS_STYLE["aborted"] = "red"`
- **Edit:** `sentinal/exceptions/runtime.py` — ExecutionAborted
- **Edit:** `examples/search_agent.py` — streaming steppable executable
- **Edit:** `sentinal/__init__.py` — export new public types
