"""Tests for the visualization event hub. No AWS, no browser."""

import asyncio
import json
import queue
import sys
import threading
import time

import numpy as np
import pytest

from router import viz


def test_json_safe_replaces_non_finite_floats_with_none():
    # float('-inf') is what select() assigns an ineligible arm's
    # selection_score. json.dumps emits a bare -Infinity, which is invalid
    # JSON and makes the browser's JSON.parse throw, killing the stream.
    out = viz.json_safe({"a": float("-inf"), "b": float("inf"), "c": float("nan")})
    assert out == {"a": None, "b": None, "c": None}
    assert "Infinity" not in json.dumps(out)
    assert "NaN" not in json.dumps(out)


def test_json_safe_converts_numpy_types():
    out = viz.json_safe(
        {"f": np.float64(1.5), "i": np.int64(3), "arr": np.array([1.0, 2.0])}
    )
    assert out == {"f": 1.5, "i": 3, "arr": [1.0, 2.0]}
    json.dumps(out)  # must not raise


def test_json_safe_survives_a_full_score_payload():
    from router import config
    from router.bandit import LinUCB

    bandit = LinUCB(arm_names=config.ARM_NAMES)
    x = np.zeros(config.CONTEXT_DIM)
    x[48] = 1.0
    costs = {a: config.fallback_cost(a) for a in config.ARM_NAMES}
    # A tiny budget and a 2-arm candidate list makes most arms ineligible,
    # so most selection_scores are -inf.
    _arm, scores = bandit.select(x, config.ARM_NAMES[:2], costs, 0.0001)
    text = json.dumps(viz.json_safe(scores))
    assert "Infinity" not in text
    assert json.loads(text)


def test_emit_stamps_a_monotonic_seq():
    obs = viz.VizObserver()
    obs.emit("one", {})
    obs.emit("two", {})
    events = obs.recent()
    assert [e["seq"] for e in events] == [1, 2]
    assert [e["event"] for e in events] == ["one", "two"]


def test_emit_puts_the_payload_under_a_payload_key():
    obs = viz.VizObserver()
    obs.emit("scored", {"depth": 0, "chosen": "nova"})
    assert obs.recent()[0]["payload"] == {"depth": 0, "chosen": "nova"}


def test_next_turn_id_is_monotonic_and_one_based():
    obs = viz.VizObserver()
    assert [obs.next_turn_id() for _ in range(3)] == [1, 2, 3]


def test_the_ring_buffer_stays_bounded():
    obs = viz.VizObserver(ring_size=5)
    for i in range(20):
        obs.emit("e", {"i": i})
    events = obs.recent()
    assert len(events) == 5
    # It keeps the NEWEST, not the oldest.
    assert [e["payload"]["i"] for e in events] == [15, 16, 17, 18, 19]


def test_recent_filters_by_since():
    obs = viz.VizObserver()
    for _ in range(5):
        obs.emit("e", {})
    assert [e["seq"] for e in obs.recent(since=3)] == [4, 5]
    assert obs.recent(since=99) == []
    assert len(obs.recent(since=0)) == 5


def test_a_subscriber_receives_emitted_events():
    obs = viz.VizObserver()
    q = obs.subscribe()
    obs.emit("hello", {"x": 1})
    got = q.get_nowait()
    assert got["event"] == "hello"
    assert got["payload"] == {"x": 1}


def test_a_full_subscriber_queue_drops_instead_of_blocking():
    # The load-bearing property: complete() runs on Starlette's threadpool, so
    # a blocking put would freeze a live Bedrock turn behind a slow browser.
    obs = viz.VizObserver(queue_size=1)
    obs.subscribe()
    start = time.perf_counter()
    for _ in range(50):
        obs.emit("e", {})
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"emit blocked for {elapsed:.2f}s"
    assert obs.stats()["dropped"] > 0
    # Dropping a subscriber's event must not drop it from the ring buffer.
    assert len(obs.recent()) == 50


def test_unsubscribe_stops_delivery():
    obs = viz.VizObserver()
    q = obs.subscribe()
    obs.unsubscribe(q)
    obs.emit("e", {})
    with pytest.raises(queue.Empty):
        q.get_nowait()
    assert obs.stats()["subscribers"] == 0


def test_unsubscribing_twice_is_not_an_error():
    obs = viz.VizObserver()
    q = obs.subscribe()
    obs.unsubscribe(q)
    obs.unsubscribe(q)


def test_subscribe_with_history_returns_the_backlog_and_registers():
    obs = viz.VizObserver()
    obs.emit("old", {})
    q, backlog = obs.subscribe_with_history(0)
    assert [e["event"] for e in backlog] == ["old"]
    assert obs.stats()["subscribers"] == 1
    obs.emit("new", {})
    assert q.get_nowait()["event"] == "new"


def test_subscribe_with_history_honours_since():
    obs = viz.VizObserver()
    for _ in range(4):
        obs.emit("e", {})
    _q, backlog = obs.subscribe_with_history(since=2)
    assert [e["seq"] for e in backlog] == [3, 4]


def test_subscribe_with_history_never_delivers_an_event_twice():
    """Registering and reading the backlog must be ONE atomic step.

    As two separate lock acquisitions, an emit landing between them went into
    both the new queue and the backlog. A duplicate is not cosmetic: the page
    renders a repeated `scored` as an extra attempt block, shifting every later
    answer by one and showing a cascade that never happened. Measured 13.6%
    duplicate rate at this switch interval before the fix.
    """
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(40):
            obs = viz.VizObserver(ring_size=500)
            stop = threading.Event()

            def emitter():
                while not stop.is_set():
                    obs.emit("scored", {"turn_id": 1})

            thread = threading.Thread(target=emitter)
            thread.start()
            try:
                for _ in range(20):
                    q, backlog = obs.subscribe_with_history(0)
                    replayed = {e["seq"] for e in backlog}
                    live = set()
                    try:
                        while True:
                            live.add(q.get_nowait()["seq"])
                    except queue.Empty:
                        pass
                    obs.unsubscribe(q)
                    overlap = replayed & live
                    assert not overlap, f"seq delivered twice: {sorted(overlap)[:5]}"
            finally:
                stop.set()
                thread.join()
    finally:
        sys.setswitchinterval(old_interval)


def test_emit_is_safe_from_many_threads():
    obs = viz.VizObserver(ring_size=1000)
    obs.subscribe()

    def worker():
        for _ in range(100):
            obs.emit("e", {})

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert obs.stats()["emitted"] == 800
    seqs = [e["seq"] for e in obs.recent()]
    assert len(seqs) == len(set(seqs)), "seq must be unique under concurrency"


def drain(agen, limit):
    """Pull up to `limit` frames from an async generator, then close it."""
    async def run():
        out = []
        async for frame in agen:
            out.append(frame)
            if len(out) >= limit:
                break
        await agen.aclose()
        return out
    return asyncio.run(run())


def test_sse_stream_yields_wellformed_frames():
    obs = viz.VizObserver()
    obs.emit("turn_start", {"query": "hi"})
    frames = drain(viz.sse_stream(obs), 1)
    assert len(frames) == 1
    assert frames[0].startswith("data: ")
    assert frames[0].endswith("\n\n")
    record = json.loads(frames[0][len("data: "):])
    assert record["event"] == "turn_start"
    assert record["seq"] == 1


def test_sse_stream_replays_history_before_live_events():
    obs = viz.VizObserver()
    obs.emit("first", {})
    obs.emit("second", {})
    frames = drain(viz.sse_stream(obs), 2)
    events = [json.loads(f[len("data: "):])["event"] for f in frames]
    assert events == ["first", "second"]


def test_sse_stream_unsubscribes_itself_when_closed():
    """The leak regression. aclose() must return the subscriber count to 0."""
    obs = viz.VizObserver()
    obs.emit("only", {})

    async def run():
        agen = viz.sse_stream(obs)
        await agen.__anext__()                      # subscribes
        assert obs.stats()["subscribers"] == 1
        await agen.aclose()                         # runs the finally
        assert obs.stats()["subscribers"] == 0

    asyncio.run(run())


def test_emit_survives_a_value_whose_str_raises():
    # A __str__ that raises would break emit()'s "never raises" contract.
    # Real case: str(exc) of arbitrary boto3/botocore exceptions.
    class BadStr:
        def __str__(self):
            raise RuntimeError("str() exploded")

    obs = viz.VizObserver()
    obs.emit("bad", {"x": BadStr()})  # must not raise
    events = obs.recent()
    assert len(events) == 1
    assert events[0]["seq"] == 1
    assert json.dumps(events[0])  # must serialize cleanly


def test_emit_survives_a_self_referential_payload():
    # A cycle in the payload would cause RecursionError without depth limiting.
    # Real case: accidental cycle in nested numpy object arrays or future payloads.
    obs = viz.VizObserver()
    d = {}
    d["self"] = d
    obs.emit("cycle", d)  # must not raise
    events = obs.recent()
    assert len(events) == 1
    assert events[0]["seq"] == 1
    assert json.dumps(events[0])  # must serialize cleanly


def test_emit_increments_seq_even_on_serialization_failure():
    # If json_safe somehow raises despite its defenses, emit must still record
    # the event with a seq so the stream stays gap-free.
    obs = viz.VizObserver()

    class BadStr:
        def __str__(self):
            raise RuntimeError("boom")

    obs.emit("first", {"x": 1})
    obs.emit("bad", {"x": BadStr()})
    obs.emit("third", {"x": 3})

    events = obs.recent()
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert obs.stats()["emitted"] == 3


def _app_with_observer():
    """An app with a stub router: no lock, no Bedrock client, no embedder.

    Reuses test_app.StubRouter -- the existing HTTP-boundary stub -- rather than
    defining a second one, and attaches an observer the way build_app does.
    """
    from fastapi.testclient import TestClient

    from router.app import build_app
    from router.tests.test_app import StubRouter

    stub = StubRouter()
    stub.observer = viz.VizObserver()
    application = build_app(router=stub, lock=False)
    return TestClient(application), application.state.viz


def test_build_app_uses_the_routers_observer():
    client, observer = _app_with_observer()
    with client:
        # build_app must adopt the injected router's observer, not create a
        # second one -- otherwise the routes serve an always-empty stream.
        observer.emit("probe", {})
        assert client.get("/viz/events").json()["stats"]["emitted"] == 1


def test_viz_index_serves_the_html():
    client, _observer = _app_with_observer()
    with client:
        response = client.get("/viz/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text.lower()


def test_viz_events_returns_buffered_events_and_stats():
    client, observer = _app_with_observer()
    observer.emit("scored", {"depth": 0})
    with client:
        response = client.get("/viz/events")
    assert response.status_code == 200
    body = response.json()
    assert [e["event"] for e in body["events"]] == ["scored"]
    assert body["stats"]["emitted"] == 1


def test_viz_stats_returns_only_the_counters():
    client, observer = _app_with_observer()
    observer.emit("scored", {"depth": 0})
    with client:
        response = client.get("/viz/stats")
    assert response.status_code == 200
    assert response.json() == {"emitted": 1, "dropped": 0, "subscribers": 0}


def test_viz_stats_does_not_serialize_the_ring():
    """The page polls this every 5s, so its body must not scale with the ring.

    Reading the counters via /viz/events?since=0 meant serializing every
    buffered event -- measured over 600 KB -- to use ~48 bytes of it, while
    holding the observer lock and a threadpool thread Router.complete shares.
    """
    client, observer = _app_with_observer()
    answer = "x" * 2000
    for i in range(300):
        observer.emit("called", {"turn_id": i, "text": answer})
    with client:
        stats = client.get("/viz/stats")
        events = client.get("/viz/events?since=0")

    assert stats.json()["emitted"] == 300
    # Constant-size regardless of ring contents, and orders of magnitude
    # smaller than the full-ring response it replaced.
    assert len(stats.content) < 200
    assert len(events.content) > 100 * len(stats.content)


def test_viz_events_honours_since():
    client, observer = _app_with_observer()
    observer.emit("a", {})
    observer.emit("b", {})
    with client:
        response = client.get("/viz/events", params={"since": 1})
    assert [e["event"] for e in response.json()["events"]] == ["b"]


def test_viz_stream_serves_sse_over_http():
    """Exercises the real route over HTTP.

    Starlette 1.3.1's TestClient buffers the whole body before returning a
    response (testclient.py:353), so an endless generator would hang forever
    -- client.stream() cannot help, the headers never arrive. Instead we let
    the stream END: a timer thread emits a live event and then unsubscribes,
    which is exactly the condition sse_stream returns on. That makes the
    buffered response complete and lets us assert on real HTTP output.
    """
    from fastapi.testclient import TestClient

    from router.app import build_app
    from router.tests.test_app import StubRouter

    stub = StubRouter()
    stub.observer = viz.VizObserver()
    app = build_app(router=stub, lock=False)
    observer = app.state.viz
    observer.emit("replayed", {"n": 1})

    def stop_the_stream():
        for _ in range(400):
            with observer._lock:
                queues = list(observer._subscribers)
            if queues:
                observer.emit("live", {"n": 2})
                time.sleep(0.05)
                observer.unsubscribe(queues[0])
                return
            time.sleep(0.01)

    with TestClient(app) as client:
        stopper = threading.Thread(target=stop_the_stream)
        stopper.start()
        response = client.get("/viz/stream")
        stopper.join()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    frames = [f for f in response.text.split("\n\n") if f.strip()]
    events = [
        json.loads(f[len("data: "):])["event"]
        for f in frames
        if f.startswith("data: ")
    ]
    assert events == ["replayed", "live"]      # ring replay, then live fan-out
    assert observer.stats()["subscribers"] == 0
