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


# Task 2: DashboardState tests
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


# Task 3: LiveThreadDashboard tests
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


# Task 4: Exports + configure(live_dashboard=...) tests
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


# Task 5: Actor instrumentation test
import asyncio as _asyncio
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


def test_observer_emits_change_detected():
    activity_bus.attach()
    try:
        obs = Observer(ActorRef("observer", "local"))

        class _S:
            class _State:
                def get_all_memory(self_inner):
                    return {"actor_status": "executing"}
            state = _State()
            session_id = "sess-xyz"
            _polls = 2
            def is_terminal(self_inner):
                self_inner._polls -= 1
                return self_inner._polls < 0

        _asyncio.run(obs.observe_session(_S(), poll_interval=0.001))
        actions = {e.action for e in activity_bus.drain()}
        assert "change.detected" in actions
        assert "observing" in actions
        assert "complete" in actions
    finally:
        activity_bus.detach()


def test_dashboard_captures_terminal_events_under_gather():
    """Integration test: proves the gather(execute, dash.run) pattern captures terminal events."""

    async def _run_test():
        bus = activity_bus

        class _FakeSession:
            def __init__(self):
                self.session_id = "sess-gather-test"
                self._polls = 0
                self.state = self
                self._terminal = False

            def get_all_memory(self):
                return {}

            def is_terminal(self):
                return self._terminal

            def mark_terminal(self):
                self._terminal = True

        async def fake_execute(session):
            """Simulates an executor emitting lifecycle events."""
            await _asyncio.sleep(0.005)  # Let dashboard attach first
            bus.emit("Executor", "started", status="running")
            await _asyncio.sleep(0.005)
            bus.emit("Executor", "executing", status="running", detail="agent")
            await _asyncio.sleep(0.005)
            bus.emit("Executor", "completed", status="completed")
            await _asyncio.sleep(0.005)
            session.mark_terminal()  # Signal terminal after all events

        session = _FakeSession()
        dash = LiveThreadDashboard(bus=bus, tick=0.001, force_plain=True)

        try:
            await _asyncio.gather(
                fake_execute(session),
                dash.run(session),
            )

            # Verify the dashboard captured the terminal "completed" event
            rows = {r.actor: r for r in dash._state.rows()}
            assert "Executor" in rows
            executor_row = rows["Executor"]
            assert executor_row.status == "completed"
            assert executor_row.action == "completed"
            assert executor_row.count == 3  # started, executing, completed
        finally:
            # Safety net: ensure bus is detached
            if bus.is_attached():
                bus.detach()

    _asyncio.run(_run_test())
