# Live Thread Dashboard — Design

**Date:** 2026-07-24
**Status:** Approved (pending spec review)
**Author:** Sentinal team

## Problem

When an agent runs through the Sentinal runtime, three actors (Executor, Observer,
Governor) execute concurrently on separate OS threads via `HybridRunner.run_actors`
(`sentinal/compute/runner.py`). Today there is no way to *see* what each thread is
doing while it runs. Tracing (`@traced`) is wired but disabled by default
(`trace_exporter="none"`) and requires an external UI (Jaeger); logs are plain text
with no thread identity. The user wants to watch, in the terminal, what each thread
is doing at each moment — concurrently, side by side.

## Goal

A live, in-terminal dashboard that renders one row per worker thread and updates in
real time as the three actors do their work, so the concurrency is directly visible.
Granularity: **actor lifecycle + key steps** (agent thinking, tool calls,
observations detected, governor decisions) — not every decorated function call, and
not just coarse phases.

## Non-Goals

- No historical/replay timeline (that is the separate OTel→Jaeger path).
- No web UI or exported HTML.
- No change to default behavior: the dashboard is opt-in; headless runs and existing
  tests are unaffected.
- No support for multiple concurrent sessions in the UI today (but the event schema
  carries `session_id` to keep that door open).

## Architecture

```
Executor thread ─┐
Observer thread ─┼─► ThreadActivityBus  ──drain──►  LiveThreadDashboard  ──► terminal
Governor thread ─┘   (thread-safe queue)   (main asyncio task, Rich Live)
```

Two new components in a new module `sentinal/telemetry/dashboard.py`, plus light
`emit` instrumentation inside the three actors and the example's `web_search` tool.

### ThreadActivityBus

- Thread-safe singleton wrapping `queue.Queue`.
- Public method: `emit(actor: str, action: str, *, detail: str | None = None,
  status: str | None = None, session_id: str | None = None)`.
- Each emitted event is a small dataclass `ThreadEvent` stamped with:
  - `thread_id` — `threading.get_native_id()` of the calling thread
  - `thread_name` — `threading.current_thread().name`
  - `actor` — logical actor name ("Executor" / "Observer" / "Governor" / "tool")
  - `action` — short verb string (e.g. `agent.thinking`, `tool.web_search`)
  - `detail` — optional short string (query, changed key, score)
  - `status` — optional status token (`running`, `observing`, `governing`,
    `completed`, `failed`)
  - `session_id` — optional, for future multi-session filtering
  - `ts` — `time.monotonic()` timestamp
- **No-op when detached:** if no dashboard has attached a sink, `emit` returns
  immediately (events dropped). Instrumentation is therefore always safe to leave in
  place; headless runs pay only a cheap branch. This mirrors the OTel `NoOp`
  degradation pattern already in `telemetry/tracing.py`.
- Attachment: `LiveThreadDashboard` registers itself as the active sink on start and
  deregisters on stop.

### LiveThreadDashboard

- Runs as an asyncio task on the main thread, concurrently with `runtime.execute()`.
- On a ~10Hz tick (`await asyncio.sleep(0.1)`): drains all queued events, folds them
  into per-thread state, and re-renders.
- Per-thread state (keyed by `thread_id`): actor name, latest status, latest action,
  latest detail, event count, last-seen timestamp.
- Also keeps a short ring buffer (last ~5) of events across all threads for the
  "recent" ticker line.
- Termination: loops until `session.is_terminal()` (same signal the Observer uses,
  `actors/observer.py:104`), then does a final render and stops the Rich `Live`
  context so the last frame stays in scrollback.

## Rendering

Rich `Live` display with a header panel and a table (one row per thread):

```
┌─ Sentinal · Live Thread View ──────────────────────────────  GIL: DISABLED ─┐
│  Session 3f9a…  ·  elapsed 2.4s  ·  backend: cpu                              │
└──────────────────────────────────────────────────────────────────────────────┘
┏━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Actor      ┃ TID   ┃ Status    ┃ Current action          ┃ Detail  ┃ Count ┃
┡━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ ● Executor │ 12841 │ running   │ tool.web_search         │ "py 3.14…"│  4   │
│ ● Observer │ 12842 │ observing │ change.detected         │ result   │ 17   │
│ ● Governor │ 12843 │ governing │ quality.score           │ 0.85     │  3   │
└────────────┴───────┴───────────┴─────────────────────────┴─────────┴───────┘
  recent ▸ [12841] agent.thinking → tool.web_search → agent.thinking
```

- One row per native thread id — three rows updating at once makes concurrency
  visible directly.
- Status dot color: green = running/active, yellow = waiting/polling,
  blue = complete, red = failed.
- Count = events seen from that thread; the Observer's fast polling visibly ticks up
  while the Executor sits on a slow LLM call.
- Header shows session id (short), elapsed time, resolved backend, and GIL state.
- Recent-events ticker shows the last few actions interleaved across threads.

## Instrumentation points

Direct `bus.emit(...)` calls added at actor lifecycle points:

**Executor** (`actors/executor.py`, runs on backend/GPU thread):
- `started` (status=running) at the top of `execute_session`
- `executing` right before `session.executable.execute(...)` (line ~133)
- `completed` / `failed` with a short result/error summary

Note: the model call and `agent.thinking` happen *inside* the user's Executable
(the LangGraph agent), not in the Executor actor. Fine-grained agent/tool steps are
emitted from the Executable/tool layer (see **Tool** below and the example), while the
Executor emits its own actor lifecycle. Both run on the same backend thread, so they
share a `thread_id` and collapse into the Executor row.

**Observer** (`actors/observer.py`, CPU thread):
- `observing` on start
- `change.detected` each time `_detect_changes` returns changes, detail = changed key
- `complete` with observation count

**Governor** (`actors/governor.py`, CPU thread):
- `governing` on start
- `quality.score` with the score value
- decision (`accept` / `intervene`) as detail

**Tool** (example `web_search` in `examples/search_agent.py`):
- one `tool.web_search` emit with the query as detail

All emits carry `session_id` where a session is in scope.

## Integration

- **Opt-in:** `configure(..., live_dashboard=True)` or env `SENTINAL_LIVE_DASHBOARD=1`.
  Default off.
- **Wiring:** the example runs the dashboard concurrently with execution, e.g.
  `asyncio.gather(dashboard.run(session), runtime.execute(session))`, or via a small
  `runtime.execute_with_dashboard(session)` convenience helper.
- **pyproject:** add a new optional-dependency group
  `dashboard = ["rich>=13.0"]` (and include `rich` in `all`).

## Error handling & degradation

- **rich not installed:** `LiveThreadDashboard` falls back to periodic plain-text
  thread-status lines (no crash), mirroring OTel NoOp degradation.
- **Non-TTY stdout (piped/CI):** dashboard auto-disables and normal logging proceeds.
  Rich's `Live` detects this; we also guard explicitly with `sys.stdout.isatty()`.
- **Bus overflow:** `queue.Queue` is unbounded for simplicity; events are drained
  every 100ms so backlog is bounded in practice. If this proves a problem, switch to a
  bounded queue that drops oldest — out of scope for v1.
- **Exceptions in dashboard task:** caught and logged; a dashboard failure must never
  crash the agent run.

## Testing

- **ThreadActivityBus:** emit/drain correctness; thread-safety under concurrent
  emits from multiple threads; no-op behavior when detached.
- **Dashboard state reduction:** feed a scripted list of `ThreadEvent`s and assert the
  resulting per-thread state and recent-ticker contents — no real terminal required.
- **Degradation:** simulate missing `rich` / non-TTY and assert graceful fallback.
- Existing runtime/actor tests must remain green (instrumentation is additive and
  no-ops when detached).

## Files touched

- **New:** `sentinal/telemetry/dashboard.py` (bus + dashboard)
- **New:** `tests/test_telemetry/test_dashboard.py`
- **Edit:** `sentinal/actors/executor.py`, `observer.py`, `governor.py` — emit calls
- **Edit:** `sentinal/telemetry/__init__.py` — export bus + dashboard;
  `configure(live_dashboard=...)`
- **Edit:** `examples/search_agent.py` — enable dashboard, emit in `web_search`
- **Edit:** `pyproject.toml` — `[dashboard]` optional dependency
