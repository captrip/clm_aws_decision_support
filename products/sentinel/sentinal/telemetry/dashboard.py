"""Live per-thread activity dashboard (opt-in).

ThreadActivityBus collects events emitted by actors on their own OS threads.
LiveThreadDashboard drains and renders them. When no dashboard is
attached, emit() is a cheap no-op — instrumentation is always safe to leave in.
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import sys
import threading
import time
from collections import deque, OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def dashboard_enabled() -> bool:
    return os.environ.get("SENTINAL_LIVE_DASHBOARD") == "1"


try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


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


_STATUS_STYLE = {
    "running": "green",
    "executing": "green",
    "observing": "yellow",
    "governing": "yellow",
    "completed": "blue",
    "complete": "blue",
    "failed": "red",
    "aborted": "red",
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
