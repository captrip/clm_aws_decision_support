# Live Thread Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in in-terminal dashboard that renders one row per worker thread (Executor/Observer/Governor) and updates in real time so concurrent actor activity is visible.

**Architecture:** A thread-safe `ThreadActivityBus` singleton collects `ThreadEvent`s emitted by actors on their own OS threads. A `LiveThreadDashboard` runs as an asyncio task on the main thread, drains the bus every ~100ms, folds events into per-thread state, and renders a Rich `Live` table. The bus is a no-op when no dashboard is attached, so instrumentation is always safe and headless runs are unaffected.

**Tech Stack:** Python 3.13+, `rich>=13.0` (new optional `[dashboard]` group, degrades to plain text if absent), stdlib `queue`/`threading`/`asyncio`, `pytest`/`pytest-asyncio`.

## Global Constraints

- Python `>=3.13` (from `pyproject.toml`).
- Core `dependencies = []` must stay empty; `rich` goes only in an optional group `dashboard = ["rich>=13.0"]` and is added to `all`.
- Dashboard is **opt-in**: default off. Existing runs and tests must remain green with no behavior change.
- `bus.emit(...)` must be a cheap no-op when no dashboard is attached (events dropped, never raises).
- A dashboard failure must never crash the agent run.
- Non-TTY stdout (piped/CI) auto-disables the live view; normal logging proceeds.
- Follow existing module style: `from __future__ import annotations`, module-level `logger = logging.getLogger(__name__)`, graceful-degradation pattern mirroring `telemetry/tracing.py` NoOp.
- Event schema fields (verbatim): `thread_id: int`, `thread_name: str`, `actor: str`, `action: str`, `detail: str | None`, `status: str | None`, `session_id: str | None`, `ts: float`.

---

### Task 1: ThreadActivityBus + ThreadEvent

**Files:**
- Create: `sentinal/telemetry/dashboard.py`
- Test: `tests/test_telemetry/test_dashboard.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces:
  - `@dataclass(frozen=True) ThreadEvent` with fields `thread_id: int`, `thread_name: str`, `actor: str`, `action: str`, `detail: str | None = None`, `status: str | None = None`, `session_id: str | None = None`, `ts: float = 0.0`.
  - `class ThreadActivityBus` with:
    - `emit(self, actor: str, action: str, *, detail: str | None = None, status: str | None = None, session_id: str | None = None) -> None` — stamps `thread_id`/`thread_name`/`ts` and enqueues **only if a sink is attached**, else returns immediately.
    - `attach(self) -> None` / `detach(self) -> None` — toggle the active sink (create/clear the internal `queue.Queue`).
    - `drain(self) -> list[ThreadEvent]` — non-blocking, returns all queued events (empty list if none/detached).
    - `is_attached(self) -> bool`.
  - Module-level singleton `activity_bus = ThreadActivityBus()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telemetry/test_dashboard.py
import threading
from sentinal.telemetry.dashboard import ThreadActivityBus, ThreadEvent


def test_emit_is_noop_when_detached():
    bus = ThreadActivityBus()
    bus.emit("Executor", "started")
    assert bus.drain() == []
    assert bus.is_attached() is False


def test_emit_and_drain_when_attached():
    bus = ThreadActivityBus()
    bus.attach()
    bus.emit("Executor", "started", status="running", detail="q")
    events = bus.drain()
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, ThreadEvent)
    assert e.actor == "Executor"
    assert e.action == "started"
    assert e.status == "running"
    assert e.detail == "q"
    assert e.thread_id == threading.get_native_id()
    assert e.ts > 0
    assert bus.drain() == []  # drain empties the queue


def test_detach_stops_collecting():
    bus = ThreadActivityBus()
    bus.attach()
    bus.detach()
    bus.emit("Observer", "observing")
    assert bus.drain() == []


def test_thread_safe_concurrent_emit():
    bus = ThreadActivityBus()
    bus.attach()

    def worker():
        for _ in range(100):
            bus.emit("Observer", "change.detected")

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(bus.drain()) == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinal.telemetry.dashboard'`

- [ ] **Step 3: Write minimal implementation**

```python
# sentinal/telemetry/dashboard.py
"""Live per-thread activity dashboard (opt-in).

ThreadActivityBus collects events emitted by actors on their own OS threads.
LiveThreadDashboard (Task 4) drains and renders them. When no dashboard is
attached, emit() is a cheap no-op — instrumentation is always safe to leave in.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreadEvent:
    thread_id: int
    thread_name: str
    actor: str
    action: str
    detail: str | None = None
    status: str | None = None
    session_id: str | None = None
    ts: float = 0.0


class ThreadActivityBus:
    """Thread-safe collector of ThreadEvents. No-op until attached."""

    def __init__(self) -> None:
        self._queue: queue.Queue[ThreadEvent] | None = None
        self._lock = threading.Lock()

    def attach(self) -> None:
        with self._lock:
            self._queue = queue.Queue()

    def detach(self) -> None:
        with self._lock:
            self._queue = None

    def is_attached(self) -> bool:
        return self._queue is not None

    def emit(
        self,
        actor: str,
        action: str,
        *,
        detail: str | None = None,
        status: str | None = None,
        session_id: str | None = None,
    ) -> None:
        q = self._queue
        if q is None:
            return
        thread = threading.current_thread()
        event = ThreadEvent(
            thread_id=threading.get_native_id(),
            thread_name=thread.name,
            actor=actor,
            action=action,
            detail=detail,
            status=status,
            session_id=session_id,
            ts=time.monotonic(),
        )
        try:
            q.put_nowait(event)
        except Exception:  # never let telemetry crash the caller
            pass

    def drain(self) -> list[ThreadEvent]:
        q = self._queue
        if q is None:
            return []
        events: list[ThreadEvent] = []
        while True:
            try:
                events.append(q.get_nowait())
            except queue.Empty:
                break
        return events


activity_bus = ThreadActivityBus()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/telemetry/dashboard.py tests/test_telemetry/test_dashboard.py
git commit -m "feat: add ThreadActivityBus + ThreadEvent for live dashboard"
```

---

### Task 2: Dashboard state reducer

**Files:**
- Modify: `sentinal/telemetry/dashboard.py`
- Test: `tests/test_telemetry/test_dashboard.py`

**Interfaces:**
- Consumes: `ThreadEvent` (Task 1).
- Produces:
  - `@dataclass ThreadRow` with `thread_id: int`, `actor: str`, `status: str`, `action: str`, `detail: str`, `count: int`.
  - `class DashboardState` with:
    - `apply(self, event: ThreadEvent) -> None` — folds one event into per-thread state; increments that thread's count; updates status/action/detail; appends to a recent ring buffer (maxlen 5).
    - `rows(self) -> list[ThreadRow]` — rows sorted by first-seen order (stable).
    - `recent(self) -> list[ThreadEvent]` — last ≤5 events across all threads.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_telemetry/test_dashboard.py
import time as _time
from sentinal.telemetry.dashboard import DashboardState, ThreadEvent


def _ev(tid, actor, action, **kw):
    return ThreadEvent(thread_id=tid, thread_name=f"t{tid}", actor=actor,
                       action=action, ts=_time.monotonic(), **kw)


def test_state_folds_events_per_thread():
    state = DashboardState()
    state.apply(_ev(1, "Executor", "started", status="running"))
    state.apply(_ev(1, "Executor", "executing", status="running", detail="agent"))
    state.apply(_ev(2, "Observer", "observing", status="observing"))

    rows = {r.thread_id: r for r in state.rows()}
    assert rows[1].actor == "Executor"
    assert rows[1].action == "executing"
    assert rows[1].detail == "agent"
    assert rows[1].count == 2
    assert rows[2].count == 1
    assert rows[2].status == "observing"


def test_recent_ring_buffer_caps_at_5():
    state = DashboardState()
    for i in range(8):
        state.apply(_ev(1, "Executor", f"step{i}"))
    recent = state.recent()
    assert len(recent) == 5
    assert recent[-1].action == "step7"


def test_rows_stable_order_by_first_seen():
    state = DashboardState()
    state.apply(_ev(3, "Governor", "governing"))
    state.apply(_ev(1, "Executor", "started"))
    state.apply(_ev(3, "Governor", "quality.score"))
    assert [r.thread_id for r in state.rows()] == [3, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -k state -v`
Expected: FAIL with `ImportError: cannot import name 'DashboardState'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to sentinal/telemetry/dashboard.py (after ThreadEvent, before ThreadActivityBus is fine;
# add these imports at top: from collections import deque, OrderedDict)

from collections import deque, OrderedDict


@dataclass
class ThreadRow:
    thread_id: int
    actor: str
    status: str
    action: str
    detail: str
    count: int


class DashboardState:
    """Folds ThreadEvents into per-thread rows + a recent ring buffer."""

    def __init__(self) -> None:
        self._rows: "OrderedDict[int, ThreadRow]" = OrderedDict()
        self._recent: deque[ThreadEvent] = deque(maxlen=5)

    def apply(self, event: ThreadEvent) -> None:
        row = self._rows.get(event.thread_id)
        if row is None:
            row = ThreadRow(
                thread_id=event.thread_id,
                actor=event.actor,
                status=event.status or "",
                action=event.action,
                detail=event.detail or "",
                count=0,
            )
            self._rows[event.thread_id] = row
        row.actor = event.actor
        row.action = event.action
        if event.status is not None:
            row.status = event.status
        if event.detail is not None:
            row.detail = event.detail
        row.count += 1
        self._recent.append(event)

    def rows(self) -> list[ThreadRow]:
        return list(self._rows.values())

    def recent(self) -> list[ThreadEvent]:
        return list(self._recent)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/telemetry/dashboard.py tests/test_telemetry/test_dashboard.py
git commit -m "feat: add DashboardState reducer for per-thread rows"
```

---

### Task 3: LiveThreadDashboard render loop (with Rich + plain-text fallback)

**Files:**
- Modify: `sentinal/telemetry/dashboard.py`
- Test: `tests/test_telemetry/test_dashboard.py`

**Interfaces:**
- Consumes: `activity_bus` (Task 1), `DashboardState` (Task 2).
- Produces:
  - `class LiveThreadDashboard`:
    - `__init__(self, bus: ThreadActivityBus | None = None, *, tick: float = 0.1, force_plain: bool | None = None)` — `bus` defaults to module `activity_bus`; `force_plain=None` auto-detects (`rich` importable AND `sys.stdout.isatty()`).
    - `async run(self, session) -> None` — attaches the bus, loops `while not session.is_terminal()`: drain → apply → render, sleeping `tick` between frames; on exit does a final drain+render and detaches the bus. Wraps the body in try/except so a render failure logs and returns without raising.
    - `_render(self) -> None` — Rich table if available, else `_render_plain()`.
    - `_render_plain(self) -> None` — prints one status line per thread.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_telemetry/test_dashboard.py
import asyncio
from sentinal.telemetry.dashboard import LiveThreadDashboard, activity_bus


class _FakeSession:
    """Terminal after N polls; emits events onto the bus as it 'runs'."""
    def __init__(self, polls, bus):
        self._left = polls
        self._bus = bus
        self._emitted = False

    def is_terminal(self):
        if not self._emitted:
            self._bus.emit("Executor", "started", status="running")
            self._bus.emit("Observer", "observing", status="observing")
            self._emitted = True
        self._left -= 1
        return self._left <= 0


def test_dashboard_runs_and_detaches(capsys):
    bus = activity_bus
    dash = LiveThreadDashboard(bus=bus, tick=0.001, force_plain=True)
    session = _FakeSession(polls=3, bus=bus)
    asyncio.run(dash.run(session))
    # bus detached after run so later emits are dropped
    assert bus.is_attached() is False
    bus.emit("Governor", "governing")
    assert bus.drain() == []


def test_dashboard_never_raises_on_render_error(capsys):
    bus = activity_bus
    dash = LiveThreadDashboard(bus=bus, tick=0.001, force_plain=True)

    class _Boom(_FakeSession):
        def is_terminal(self):
            raise RuntimeError("boom")

    # Should swallow the error, not propagate
    asyncio.run(dash.run(_Boom(polls=1, bus=bus)))
    assert bus.is_attached() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -k dashboard -v`
Expected: FAIL with `ImportError: cannot import name 'LiveThreadDashboard'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to sentinal/telemetry/dashboard.py
# top-of-file additions: import asyncio, import sys

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


_STATUS_STYLE = {
    "running": "green",
    "executing": "green",
    "observing": "yellow",
    "governing": "yellow",
    "completed": "blue",
    "complete": "blue",
    "failed": "red",
}


class LiveThreadDashboard:
    def __init__(self, bus: "ThreadActivityBus | None" = None, *, tick: float = 0.1,
                 force_plain: bool | None = None) -> None:
        self._bus = bus if bus is not None else activity_bus
        self._tick = tick
        self._state = DashboardState()
        if force_plain is None:
            force_plain = not (_RICH_AVAILABLE and sys.stdout.isatty())
        self._plain = force_plain

    async def run(self, session) -> None:
        self._bus.attach()
        try:
            if self._plain:
                await self._run_plain(session)
            else:
                await self._run_rich(session)
        except Exception as ex:  # dashboard must never crash the run
            logger.warning(f"Live dashboard stopped: {ex}")
        finally:
            self._drain_apply()
            self._bus.detach()

    def _drain_apply(self) -> None:
        for event in self._bus.drain():
            self._state.apply(event)

    async def _run_plain(self, session) -> None:
        while not session.is_terminal():
            self._drain_apply()
            self._render_plain()
            await asyncio.sleep(self._tick)
        self._drain_apply()
        self._render_plain()

    async def _run_rich(self, session) -> None:
        console = Console()
        with Live(self._render_table(), console=console, refresh_per_second=10) as live:
            while not session.is_terminal():
                self._drain_apply()
                live.update(self._render_table())
                await asyncio.sleep(self._tick)
            self._drain_apply()
            live.update(self._render_table())

    def _render_table(self):
        table = Table(title="Sentinal · Live Thread View")
        table.add_column("Actor")
        table.add_column("TID")
        table.add_column("Status")
        table.add_column("Current action")
        table.add_column("Detail")
        table.add_column("Count", justify="right")
        for row in self._state.rows():
            style = _STATUS_STYLE.get(row.status, "white")
            table.add_row(
                f"[{style}]●[/] {row.actor}",
                str(row.thread_id),
                row.status,
                row.action,
                (row.detail[:24] + "…") if len(row.detail) > 25 else row.detail,
                str(row.count),
            )
        recent = " → ".join(e.action for e in self._state.recent())
        return Panel(table, subtitle=f"recent ▸ {recent}") if recent else table

    def _render_plain(self) -> None:
        parts = [
            f"[{r.actor} tid={r.thread_id} {r.status}] {r.action} "
            f"({r.detail}) x{r.count}"
            for r in self._state.rows()
        ]
        if parts:
            print(" | ".join(parts))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/telemetry/dashboard.py tests/test_telemetry/test_dashboard.py
git commit -m "feat: add LiveThreadDashboard render loop with plain-text fallback"
```

---

### Task 4: Export dashboard + configure(live_dashboard=...)

**Files:**
- Modify: `sentinal/telemetry/__init__.py`
- Modify: `sentinal/__init__.py`
- Test: `tests/test_telemetry/test_dashboard.py`

**Interfaces:**
- Consumes: `activity_bus`, `LiveThreadDashboard` (Tasks 1–3).
- Produces:
  - `configure(..., live_dashboard: bool = False)` sets `os.environ["SENTINAL_LIVE_DASHBOARD"] = "1"` when True.
  - `dashboard_enabled() -> bool` in `sentinal.telemetry.dashboard` — True if `SENTINAL_LIVE_DASHBOARD == "1"`.
  - `sentinal.telemetry` exports `activity_bus`, `LiveThreadDashboard`, `dashboard_enabled`.
  - `sentinal` top-level exports `activity_bus`, `LiveThreadDashboard`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_telemetry/test_dashboard.py
import os
from sentinal.telemetry.dashboard import dashboard_enabled
from sentinal.telemetry import configure


def test_configure_enables_dashboard_flag(monkeypatch):
    monkeypatch.delenv("SENTINAL_LIVE_DASHBOARD", raising=False)
    assert dashboard_enabled() is False
    configure(trace_exporter="none", live_dashboard=True)
    assert os.environ["SENTINAL_LIVE_DASHBOARD"] == "1"
    assert dashboard_enabled() is True


def test_top_level_exports():
    import sentinal
    assert hasattr(sentinal, "activity_bus")
    assert hasattr(sentinal, "LiveThreadDashboard")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -k "configure or export" -v`
Expected: FAIL with `ImportError: cannot import name 'dashboard_enabled'`

- [ ] **Step 3: Write minimal implementation**

In `sentinal/telemetry/dashboard.py`, add near the top (after imports):

```python
import os


def dashboard_enabled() -> bool:
    return os.environ.get("SENTINAL_LIVE_DASHBOARD") == "1"
```

In `sentinal/telemetry/__init__.py`, change the `configure` signature and body:

```python
def configure(
    service_name: str = "sentinal",
    log_level: str = "INFO",
    log_format: str = "text",
    trace_exporter: str = "auto",
    live_dashboard: bool = False,
) -> None:
```

Add before `setup_logging(...)` at the end of the function body:

```python
    if live_dashboard:
        os.environ["SENTINAL_LIVE_DASHBOARD"] = "1"
```

Add imports + exports in `sentinal/telemetry/__init__.py`:

```python
from sentinal.telemetry.dashboard import (
    activity_bus,
    LiveThreadDashboard,
    dashboard_enabled,
)
```

and extend `__all__` with `"activity_bus"`, `"LiveThreadDashboard"`, `"dashboard_enabled"`.

In `sentinal/__init__.py`, extend the telemetry import and `__all__`:

```python
from sentinal.telemetry import traced, logged, configure, activity_bus, LiveThreadDashboard
```

add `"activity_bus"` and `"LiveThreadDashboard"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/telemetry/__init__.py sentinal/__init__.py sentinal/telemetry/dashboard.py tests/test_telemetry/test_dashboard.py
git commit -m "feat: export dashboard + configure(live_dashboard) flag"
```

---

### Task 5: Instrument the three actors with emit calls

**Files:**
- Modify: `sentinal/actors/executor.py`
- Modify: `sentinal/actors/observer.py`
- Modify: `sentinal/actors/governor.py`
- Test: `tests/test_telemetry/test_dashboard.py`

**Interfaces:**
- Consumes: `activity_bus` (Task 1).
- Produces: emit calls at documented lifecycle points. No signature changes to actor methods.

Emit points (all use `from sentinal.telemetry.dashboard import activity_bus`, `sid = str(session.session_id)`):

- **Executor.execute_session:** after "starting" memory update → `activity_bus.emit("Executor", "started", status="running", session_id=sid)`; immediately before `result = await session.executable.execute(...)` → `activity_bus.emit("Executor", "executing", status="running", session_id=sid)`; after success → `activity_bus.emit("Executor", "completed", status="completed", session_id=sid)`; in the `except Exception` block → `activity_bus.emit("Executor", "failed", status="failed", detail=str(ex), session_id=sid)`.
- **Observer.observe_session:** at start → `activity_bus.emit("Observer", "observing", status="observing", session_id=sid)`; inside `if changes:` for each key → `activity_bus.emit("Observer", "change.detected", status="observing", detail=key, session_id=sid)`; after loop → `activity_bus.emit("Observer", "complete", status="complete", detail=str(observation_count), session_id=sid)`.
- **Governor.govern_session:** at start → `activity_bus.emit("Governor", "governing", status="governing", session_id=sid)`; in `_check_quality` after computing quality → `activity_bus.emit("Governor", "quality.score", status="governing", detail=f"{quality:.2f}", session_id=sid)` (pass `session` is already in scope there).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_telemetry/test_dashboard.py
import asyncio as _asyncio
from sentinal.telemetry.dashboard import activity_bus
from sentinal.core.actor_ref import ActorRef
from sentinal.actors.observer import Observer


def test_observer_emits_lifecycle_events():
    activity_bus.attach()
    try:
        obs = Observer(ActorRef("observer", "local"))

        class _S:
            class _State:
                def get_all_memory(self_inner):
                    return {}
            state = _State()
            session_id = "sess-xyz"
            _polls = 2
            def is_terminal(self_inner):
                self_inner._polls -= 1
                return self_inner._polls < 0

        _asyncio.run(obs.observe_session(_S(), poll_interval=0.001))
        actions = {e.action for e in activity_bus.drain()}
        assert "observing" in actions
        assert "complete" in actions
    finally:
        activity_bus.detach()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -k observer_emits -v`
Expected: FAIL — `assert "observing" in actions` fails (empty set) because no emits yet.

- [ ] **Step 3: Write minimal implementation**

In `sentinal/actors/observer.py`, add import at top: `from sentinal.telemetry.dashboard import activity_bus`. In `observe_session`, right after the opening `logger.info(...)`:

```python
        sid = str(session.session_id)
        activity_bus.emit("Observer", "observing", status="observing", session_id=sid)
```

Inside `if changes:` loop (after appending each observation):

```python
                    activity_bus.emit("Observer", "change.detected",
                                      status="observing", detail=key, session_id=sid)
```

After the `while` loop, before the final `logger.info`:

```python
        activity_bus.emit("Observer", "complete", status="complete",
                          detail=str(observation_count), session_id=sid)
```

Apply the analogous emit calls to `executor.py` and `governor.py` as listed in the Interfaces block above (import `activity_bus` at top of each; compute `sid = str(session.session_id)` / `str(session.execution_id)` where a session is in scope — use `session.session_id`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.ft-env/bin/python -m pytest tests/test_telemetry/test_dashboard.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add sentinal/actors/executor.py sentinal/actors/observer.py sentinal/actors/governor.py tests/test_telemetry/test_dashboard.py
git commit -m "feat: instrument actors with ThreadActivityBus emit calls"
```

---

### Task 6: pyproject [dashboard] optional dependency

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: installable extra `sentinal[dashboard]`.

- [ ] **Step 1: Add the optional dependency group**

In `pyproject.toml` under `[project.optional-dependencies]`, add:

```toml
dashboard = ["rich>=13.0"]
```

and add `"rich>=13.0"` to the `all` list.

- [ ] **Step 2: Install and verify import**

Run: `.ft-env/bin/pip install 'rich>=13.0' && .ft-env/bin/python -c "import rich; from sentinal.telemetry.dashboard import LiveThreadDashboard; print('ok', rich.__version__)"`
Expected: prints `ok <version>`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add [dashboard] optional dependency (rich)"
```

---

### Task 7: Wire dashboard into the search agent example

**Files:**
- Modify: `examples/search_agent.py`

**Interfaces:**
- Consumes: `LiveThreadDashboard`, `activity_bus`, `dashboard_enabled`, `configure(live_dashboard=...)`.
- Produces: example runs the dashboard concurrently with `runtime.execute(session)` when enabled; `web_search` emits a `tool.web_search` event.

- [ ] **Step 1: Enable the flag in configure()**

Change the `configure(...)` call (lines ~57–62) to add `live_dashboard=True` and set `trace_exporter="none"` unchanged. Update the `import` line to include `LiveThreadDashboard`, `activity_bus`, `dashboard_enabled`:

```python
from sentinal import (
    DevicePreference,
    HybridRunner,
    configure,
    traced,
    logged,
    LiveThreadDashboard,
    activity_bus,
)
```

- [ ] **Step 2: Emit from the web_search tool**

Inside `web_search`, after computing `results` and before building `output`, add:

```python
    activity_bus.emit("tool", "tool.web_search", detail=query[:24])
```

(Import at top of file: `from sentinal.telemetry.dashboard import activity_bus` — or reuse the top-level import.)

- [ ] **Step 3: Run execution concurrently with the dashboard**

In `main()`, replace the `result = await runtime.execute(session)` block with:

```python
    dash = LiveThreadDashboard()
    t0 = time.perf_counter()
    result, _ = await asyncio.gather(
        runtime.execute(session),
        dash.run(session),
    )
    elapsed = time.perf_counter() - t0
```

- [ ] **Step 4: Smoke-test the example (headless-safe)**

Run: `SENTINAL_LIVE_DASHBOARD=1 .ft-env/bin/python -c "import examples.search_agent as s; print('imports ok')"`
Expected: `imports ok` (no runtime errors on import). A full run requires LLM credentials; the dashboard auto-falls to plain text when stdout is not a TTY.

- [ ] **Step 5: Commit**

```bash
git add examples/search_agent.py
git commit -m "feat: wire live thread dashboard into search agent example"
```

---

### Task 8: Full suite green + README note

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Run the whole test suite**

Run: `.ft-env/bin/python -m pytest -q`
Expected: all tests pass (pre-existing + 12 new dashboard tests).

- [ ] **Step 2: Add a short README section**

Add a "Live thread dashboard" subsection documenting: enable via `configure(live_dashboard=True)` or `SENTINAL_LIVE_DASHBOARD=1`, install `pip install sentinal[dashboard]`, and that it renders one row per worker thread and degrades to plain text without `rich`/TTY.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document live thread dashboard usage"
```

---

## Self-Review

**Spec coverage:**
- ThreadActivityBus (schema, no-op-when-detached, thread-safe) → Task 1 ✓
- Dashboard state reduction (per-thread rows, recent ticker) → Task 2 ✓
- LiveThreadDashboard render loop, terminate on `is_terminal`, final frame, never-crash → Task 3 ✓
- Rich table layout + status colors + recent ticker → Task 3 ✓
- rich-missing / non-TTY degradation → Task 3 (`force_plain` auto-detect) + Task 6 ✓
- Opt-in via configure/env → Task 4 ✓
- Actor instrumentation (Executor/Observer/Governor + tool) → Task 5 + Task 7 ✓
- pyproject `[dashboard]` group → Task 6 ✓
- Example integration (`asyncio.gather`) → Task 7 ✓
- Testing (bus, state, degradation) → Tasks 1–5 ✓
- session_id on events → Task 1 schema + Task 5 emits ✓

**Placeholder scan:** No TBD/TODO/"add error handling"; all code steps show full code. ✓

**Type consistency:** `emit(...)` keyword-only signature identical across Tasks 1/5/7; `ThreadEvent`/`ThreadRow` field names consistent; `dashboard_enabled`/`activity_bus`/`LiveThreadDashboard` names consistent across Tasks 3/4/5/7. ✓

Note: `agent.thinking` fine-grained emit from inside the LangGraph agent is out of scope for v1 (the Executor emits `executing`; the tool emits `tool.web_search`). This matches the spec's clarification that agent-internal steps come from the Executable/tool layer.
