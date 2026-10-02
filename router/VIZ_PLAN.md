# Router Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live browser view of one routing turn: per-attempt arm scores, the verdict, and the LinUCB before/after for the arm that was updated.

**Architecture:** An optional `observer` on `policy.Router` emits seven events from points where `complete()` already holds the data. `router/viz.py` fans them out to SSE subscribers through bounded, drop-on-full queues plus a ring buffer for late joiners. `router/static/viz.html` renders the cascade as a node-link graph with a card per arm and per update. No build step, no npm, no CDN.

**Tech Stack:** Python 3.13, FastAPI 0.141.1 / Starlette 1.3.1, numpy, `queue.Queue`, `collections.deque`, `threading.Lock`, vanilla JS + SVG/CSS.

**Spec:** [VIZ_DESIGN.md](VIZ_DESIGN.md). Router design: [DESIGN.md](DESIGN.md).

## Global Constraints

- **Never commit anything under `router/`.** It is excluded via `.git/info/exclude:9`, not `.gitignore`. Every task ends by verifying `git status --short router/` prints nothing. Do not `git add` or `git commit` any file in this plan, and do not move the rule into `.gitignore`.
- **A visualization must never break a turn.** Every observer call site is wrapped so any exception is swallowed. `Router._emit` is the only integration point into the cascade.
- **`observer=None` must be byte-identical to today's behavior**, including doing no extra computation.
- **Never block `complete()`.** Subscriber queues are bounded and drop on full. `complete()` runs on Starlette's threadpool; a blocking `put` would freeze a live Bedrock turn behind a slow browser.
- **Every bandit before/after is computed on the vector that arm was selected with**, never the turn's final vector. `complete()` rebuilds `x` per attempt (DESIGN.md §6).
- **All event payloads must be strict-JSON safe.** `selection_score` is `float("-inf")` for ineligible arms; `json.dumps` emits bare `-Infinity`, which `JSON.parse` rejects — it would kill the stream on the first budget-pruned turn.
- **Existing tests must keep passing:** `router/.venv/bin/pytest router/tests -q` → 170 passed, 3 skipped. The 3 skips are AWS-gated (`ROUTER_SMOKE=1`) and stay skipped.
- **A running dev server makes one pre-existing test fail, and it is not your bug.**
  `test_the_default_lock_blocks_a_second_holder_from_another_cwd` acquires the *real*
  `config.LOCK_PATH`, so if a `uvicorn router.app:...` process is up it fails with
  `BlockingIOError: [Errno 35]` before this plan touches anything. Check with
  `lsof router/state/router.lock`; stop the server (plain `kill`, not `-9`, so the lifespan
  persists the snapshots) and re-run. Do not "fix" this test.
- **No new third-party dependencies.** `router/requirements.txt` is unchanged.
- **`GET /viz/*` has no auth**, consistent with `/admin/judge` and DESIGN.md §9.
- Run every command from the repo root `/Users/trips/Desktop/Online_Multi_LLM`, using `router/.venv/bin/python` and `router/.venv/bin/pytest`.

## File Structure

| File | Responsibility |
| :--- | :--- |
| Create `router/viz.py` | `json_safe()`, `VizObserver` (stamp, ring buffer, fan-out, subscribe/unsubscribe, `recent`), `sse_stream()` |
| Create `router/static/viz.html` | The whole UI: submit form, SSE client, graph render. Self-contained |
| Create `router/tests/test_viz.py` | Hub, JSON safety, routes |
| Modify `router/policy.py` | `observer=None` param, `_emit`, 7 emit sites |
| Modify `router/policy.py` tests → `router/tests/test_policy.py` | Observer equivalence, raising observer, event sequences |
| Modify `router/app.py` | Build `VizObserver`, wire to router, add 3 `/viz` routes |
| Modify `router/README.md` | Document `/viz/` and the real-spend warning |

Task order is deliberate: the hub (Task 1) has no dependencies and is pure logic; `policy.py`
(Tasks 2–3) is the highest-risk edit and gets its own review gate; routes (Task 4) need the hub;
the UI (Task 5) needs a live event stream to develop against.

---

### Task 1: The event hub — `viz.py`

**Files:**
- Create: `router/viz.py`
- Create: `router/tests/test_viz.py`

**Interfaces:**
- Consumes: nothing. Pure stdlib + numpy.
- Produces:
  - `json_safe(value, _depth=0)` → recursively convert to strict-JSON-safe types. `float('-inf')`/`inf`/`nan` → `None`; numpy scalars → Python scalars; numpy arrays → lists; dict keys → `str`. Never raises: depth-capped at `MAX_DEPTH = 32` against cycles, and its `str()` fallback is exception-guarded.
  - `VizObserver(ring_size=500, queue_size=256)`
  - `.emit(event: str, payload: dict) -> None` — stamps `seq`, appends to ring, fans out. Never raises, never blocks.
  - `.next_turn_id() -> int` — monotonic, 1-based.
  - `.subscribe() -> queue.Queue` / `.unsubscribe(q) -> None`
  - `.recent(since: int = 0) -> list[dict]` — buffered events with `seq > since`
  - `.stats() -> dict` with keys `emitted`, `dropped`, `subscribers`
  - `sse_stream(observer, since=0, poll_interval=0.05, keepalive_after=1.0)` → **async** generator
    yielding `data: {json}\n\n` strings, replaying `recent(since)` first. It calls `subscribe()`
    and `unsubscribe()` itself; callers must do neither.

    **Corrected during Task 4 (was `sse_frames(observer, q, since=0)`, a sync generator).** The
    sync version leaked a subscriber on every client disconnect, permanently. uvicorn advertises
    ASGI `spec_version: "2.3"`, so `StreamingResponse` takes the task-group branch at
    `starlette/responses.py:273` and *cancels* the streaming task on `http.disconnect`. A sync
    generator runs via `iterate_in_threadpool`, so the cancellation lands on the `await` and
    cannot be injected into the worker thread: the generator is abandoned mid-`yield` and its
    `finally` never runs. Measured against real uvicorn, `subscribers` stayed at 1 forever while
    the orphaned queue kept accumulating every event (qsize 1→2→3→4). An async generator receives
    the cancellation at an `await`, so its `finally` executes. Verified after the fix: 5 sequential
    reloads and 8 concurrent tabs all return `subscribers` to 0, and 8/8 tabs receive live fan-out.

    It polls with `get_nowait()` + `anyio.sleep` rather than a blocking `get(timeout=...)` in a
    worker thread, because a blocking get would hold one of Starlette's ~40 threadpool threads per
    open browser tab — the same pool `Router.complete` runs on. Measured: 8 concurrent streams,
    `threading.active_count()` never rose above 2. `anyio` is not a new dependency; `starlette`
    already requires `anyio<5,>=3.6.2`.

- [ ] **Step 1: Write the failing tests**

Create `router/tests/test_viz.py`:

```python
"""Tests for the visualization event hub. No AWS, no browser."""

import json
import queue
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
```

These three replaced the original two `sse_frames` tests when Task 4 corrected the generator to
async. The module needs `import asyncio`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `router/.venv/bin/pytest router/tests/test_viz.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.viz'`

- [ ] **Step 3: Write the implementation**

Create `router/viz.py`:

```python
"""Event hub for the live routing visualization. See VIZ_DESIGN.md section 4.

Two rules shape this module, both about not damaging the thing being observed:

  - emit() never blocks. Router.complete() runs on Starlette's threadpool, so a
    blocking put onto a full subscriber queue would freeze a live Bedrock turn
    behind a browser that stopped reading. Queues are bounded and drop.
  - emit() never raises. policy.Router._emit swallows exceptions as a second
    layer, but a hub that throws on ordinary conditions would make that guard do
    the real work and hide bugs.

In-process state, so this inherits the single-worker invariant (DESIGN.md
section 3) and adds a reason for it: two workers would each serve a partial
stream.
"""

import json
import math
import queue
import threading
from collections import deque

import numpy as np

RING_SIZE = 500
QUEUE_SIZE = 256
MAX_DEPTH = 32


def json_safe(value, _depth=0):
    """Convert to something json.dumps emits as strict, parseable JSON.

    bandit.select assigns an ineligible arm selection_score = float("-inf"),
    and json.dumps writes that as a bare -Infinity token. That is invalid JSON:
    the browser's JSON.parse throws and the whole event stream dies. Since
    VIZ_DESIGN.md section 5 requires showing ineligible arms, the first
    budget-pruned turn would break the UI. Non-finite floats become null.

    numpy types are handled here too -- json.dumps raises TypeError on
    np.float64, and the score payload is full of them.

    Two hostile-input guards, because emit() promises never to raise and this
    function is the only thing in it that runs arbitrary user code: the depth cap
    stops a self-referential payload from raising RecursionError, and the
    try/except around the str() fallback stops a value whose __str__ raises from
    propagating. The `called` event carries str(exc) of arbitrary botocore
    exceptions and raw model output, so neither input is hypothetical.
    """
    if _depth > MAX_DEPTH:
        return "<max depth>"
    if isinstance(value, dict):
        return {
            str(k): json_safe(v, _depth=_depth + 1) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(v, _depth=_depth + 1) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v, _depth=_depth + 1) for v in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item(), _depth=_depth + 1)
    # bool before int/float: isinstance(True, int) is True in Python, so the
    # int branch would swallow bools and emit 1/0 instead of true/false.
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str)) or value is None:
        return value
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - a __str__ that raises must not escape
        # Type name only: repr() can fail for the same reason str() just did.
        return "<unserializable %s>" % type(value).__name__


class VizObserver:
    def __init__(self, ring_size=RING_SIZE, queue_size=QUEUE_SIZE):
        self.queue_size = queue_size
        self._lock = threading.Lock()
        self._ring = deque(maxlen=ring_size)
        self._subscribers = []
        self._seq = 0
        self._turn_id = 0
        self._dropped = 0

    def next_turn_id(self):
        with self._lock:
            self._turn_id += 1
            return self._turn_id

    def emit(self, event, payload):
        try:
            safe_payload = json_safe(payload or {})
        except Exception as exc:  # noqa: BLE001 - emit() must never raise
            # Emit the event anyway, with the failure named. Dropping it would
            # advance _seq with nothing in the ring, leaving a permanent hole in
            # the sequence.
            #
            # (The original text here claimed the browser "keeps re-requesting"
            # such a gap via ?since=N. That was wrong: the page advances
            # highestSeq past any gap, so a missing seq is never re-requested --
            # it is silently absent, which is why the status line now reports a
            # per-tab dropped count inferred from seq gaps.)
            safe_payload = {"_serialization_error": type(exc).__name__}
        record = {"event": str(event), "payload": safe_payload}
        with self._lock:
            self._seq += 1
            record["seq"] = self._seq
            self._ring.append(record)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(record)
            except queue.Full:
                # Drop, never block: see the module docstring. The ring buffer
                # still has it, so /viz/events?since=N can recover it.
                with self._lock:
                    self._dropped += 1

    def subscribe(self):
        q = queue.Queue(maxsize=self.queue_size)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def recent(self, since=0):
        with self._lock:
            return [e for e in self._ring if e["seq"] > int(since)]

    def stats(self):
        with self._lock:
            return {
                "emitted": self._seq,
                "dropped": self._dropped,
                "subscribers": len(self._subscribers),
            }


async def sse_stream(observer, since=0, poll_interval=0.05, keepalive_after=1.0):
    """Yield SSE frames for one client: replay from the ring, then stream live.

    An ASYNC generator, and that is load-bearing. uvicorn advertises ASGI
    spec_version 2.3, so StreamingResponse cancels the streaming task on
    client disconnect (starlette/responses.py:273). A cancel cannot be
    injected into a threadpool worker, so a SYNC generator is abandoned mid
    -yield and its finally never runs -- leaking a subscriber whose queue
    then accumulates every event for the life of the process. An async
    generator receives the cancellation at an await point, so the finally
    below actually executes. Verified against uvicorn + real HTTP.

    Polls the queue with get_nowait + anyio.sleep rather than a blocking
    get(timeout=...) in a worker thread: a blocking get would hold one of
    Starlette's ~40 threadpool threads per open browser tab, and that is the
    same pool Router.complete runs on. Measured: 8 concurrent streams, zero
    extra threads.

    Subscribes and unsubscribes itself. The caller must not do either, or the
    queue outlives the generator and the leak returns.
    """
    q = observer.subscribe()
    try:
        for record in observer.recent(since):
            yield "data: %s\n\n" % json.dumps(record)
        idle = 0.0
        while True:
            try:
                record = q.get_nowait()
            except queue.Empty:
                await anyio.sleep(poll_interval)
                idle += poll_interval
                if idle >= keepalive_after:
                    idle = 0.0
                    with observer._lock:
                        still_subscribed = q in observer._subscribers
                    if not still_subscribed:
                        return
                    yield ": keepalive\n\n"
                continue
            idle = 0.0
            yield "data: %s\n\n" % json.dumps(record)
    finally:
        observer.unsubscribe(q)
```

The `still_subscribed` check plus `return` is what lets a caller terminate the generator
deterministically from outside by calling `observer.unsubscribe(q)` — the mechanism both the
route test and any future shutdown path rely on.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `router/.venv/bin/pytest router/tests/test_viz.py -q`
Expected: PASS, 15 tests.

- [ ] **Step 5: Confirm nothing regressed and nothing is staged**

Run: `router/.venv/bin/pytest router/tests -q`
Expected: 170 pre-existing tests still pass, plus the ~15 new `test_viz.py` tests. No
previously-passing test may fail.

Run: `git status --short router/`
Expected: **no output.** Do not commit. If anything prints, stop and report it.

---

### Task 2: The observer hook in `policy.py`

The highest-risk task: `policy.py` is the most heavily reviewed file in the router. This task
adds the hook and proves it is inert, and deliberately stops short of the event payloads
(Task 3) so a reviewer can gate the two separately.

**Files:**
- Modify: `router/policy.py`
- Test: `router/tests/test_policy.py`

**Interfaces:**
- Consumes: `viz.VizObserver` from Task 1 (only in tests; `policy.py` must **not** import `viz` — the hook is duck-typed so `policy` keeps no dependency on the visualization).
- Produces:
  - `Router.__init__(..., verify_fn=verify.check, observer=None)` — `observer` is the **last** parameter, so every existing positional call is unaffected.
  - `Router._emit(self, event, **payload) -> None` — no-op when `observer is None`; swallows every exception otherwise.
  - `self.observer` attribute.

- [ ] **Step 1: Write the failing tests**

First, extend the **existing** helper at `router/tests/test_policy.py:30`. Its real signature is
`make_router(converse_fn, verify_fn=None, tmp_path=None, forced_arms=None)` and it returns
`(router, bandit, store)`. Add one keyword parameter and forward it — do not duplicate the helper:

```python
def make_router(
    converse_fn, verify_fn=None, tmp_path=None, forced_arms=None, observer=None
):
```

and pass `observer=observer` into the `Router(...)` call it already builds.

Then append to `router/tests/test_policy.py`:

```python
class _RecordingObserver:
    """Captures events. next_turn_id mirrors viz.VizObserver's contract."""

    def __init__(self):
        self.events = []
        self._turn = 0

    def next_turn_id(self):
        self._turn += 1
        return self._turn

    def emit(self, event, payload):
        self.events.append((event, payload))

    def names(self):
        return [name for name, _ in self.events]

    def payload(self, name):
        for event_name, payload in self.events:
            if event_name == name:
                return payload
        raise AssertionError(f"no {name} event in {self.names()}")


class _ExplodingObserver:
    def next_turn_id(self):
        raise RuntimeError("boom in next_turn_id")

    def emit(self, event, payload):
        raise RuntimeError("boom in emit")


def _ok_converse(client, model_id, prompt, **kwargs):
    return "an answer", usage()


def test_emit_is_a_noop_without_an_observer(tmp_path):
    router, _bandit, store = make_router(_ok_converse, tmp_path=tmp_path)
    assert router.observer is None
    router._emit("anything", a=1)  # must not raise
    store.close()


def test_an_observer_receives_events(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    router._emit("ping", a=1)
    assert observer.events == [("ping", {"a": 1})]
    store.close()


def test_an_exploding_observer_cannot_break_emit(tmp_path):
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=_ExplodingObserver()
    )
    router._emit("ping", a=1)  # must not raise
    store.close()


def test_an_exploding_observer_cannot_break_a_turn(tmp_path):
    # The load-bearing test of this whole feature. A visualization that can take
    # down routing is worse than no visualization. Note the observer raises from
    # next_turn_id too, so the turn_id lookup must be guarded as well as _emit.
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path,
        observer=_ExplodingObserver(),
    )
    result = router.complete("conv", "q", task_type="code")
    assert result["status"] == "success"
    assert result["arm"] is not None
    store.close()


def test_observer_none_leaves_the_result_unchanged(tmp_path):
    # Same stubs, fresh identical bandits, with and without an observer:
    # identical results.
    plain, _b1, store1 = make_router(
        _ok_converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    plain_result = plain.complete("conv", "q", task_type="code")
    store1.close()

    watched, _b2, store2 = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path / "watched",
        observer=_RecordingObserver(),
    )
    watched_result = watched.complete("conv", "q", task_type="code")
    store2.close()

    for key in ("arm", "status", "cascade_depth", "verdict", "text"):
        assert plain_result[key] == watched_result[key], key
```

`make_router` builds its `Store` at `tmp_path / "policy.db"`, so the second router in the last
test needs its own directory: create it with `(tmp_path / "watched").mkdir()` before the call, or
the `Store` open will fail on the missing parent.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `router/.venv/bin/pytest router/tests/test_policy.py -q -k "observer or emit"`
Expected: FAIL — `TypeError: Router.__init__() got an unexpected keyword argument 'observer'`

- [ ] **Step 3: Write the implementation**

In `router/policy.py`, add `observer=None` as the **last** parameter of `__init__`, assign
`self.observer = observer`, and add this method directly after `__init__`:

```python
    def _emit(self, event, **payload):
        """Fire a visualization event. Never let it affect the turn.

        A visualization that can break routing is worse than no visualization:
        the observer is a debugging aid, the turn is the product. Every failure
        mode -- a full queue, a disconnected browser, a bug in viz.py -- is
        swallowed here. The observer is duck-typed so policy.py keeps no
        dependency on the visualization module.
        """
        if self.observer is None:
            return
        try:
            self.observer.emit(event, payload)
        except Exception:  # noqa: BLE001 - deliberate, see docstring
            pass
```

Do not add any emit call sites in this task — that is Task 3.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `router/.venv/bin/pytest router/tests/test_policy.py -q`
Expected: PASS, including all pre-existing tests in the file.

- [ ] **Step 5: Confirm nothing regressed and nothing is staged**

Run: `router/.venv/bin/pytest router/tests -q`
Expected: all pre-existing tests still pass, plus the 5 new observer tests. No previously-passing
test may fail.

Run: `git status --short router/`
Expected: **no output.** Do not commit.

---

### Task 3: Emit the seven events

**Files:**
- Modify: `router/policy.py`
- Test: `router/tests/test_policy.py`

**Interfaces:**
- Consumes: `Router._emit` and `self.observer` from Task 2.
- Produces: seven event names with the payload keys below. Task 5's UI reads these exact keys.

| Event | Payload keys |
| :--- | :--- |
| `turn_start` | `turn_id`, `conversation_id`, `query`, `task_type`, `budget`, `remaining` |
| `scored` | `turn_id`, `depth`, `chosen`, `remaining`, `scores` (arm → dict) |
| `called` | `turn_id`, `depth`, `arm`, `text`, `cost`, `usage`, `error`, `latency_ms` |
| `verified` | `turn_id`, `depth`, `arm`, `verdict`, `learn`, `task_type` |
| `updated` | `turn_id`, `depth`, `arm`, `reward`, `pulls_before`, `pulls_after`, `before`, `after` |
| `turn_end` | `turn_id`, `status`, `cascade_depth`, `cost`, `decision_ids` |
| `delayed_update` | `decision_id`, `arm`, `reward`, `applied`, `pulls_before`, `pulls_after`, `before`, `after` |

`before` and `after` are each `{"p": float, "e": float, "ucb": float}`.

- [ ] **Step 1: Write the failing tests**

Append to `router/tests/test_policy.py`:

```python
def test_a_verified_turn_emits_the_full_event_sequence(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path,
        observer=observer,
    )
    router.complete("conv", "q", task_type="code")
    assert observer.names() == [
        "turn_start",
        "scored",
        "called",
        "verified",
        "updated",
        "turn_end",
    ]
    store.close()


def test_scored_carries_every_arm_including_ineligible_ones(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path,
        observer=observer,
    )
    router.complete("conv", "q", task_type="code")
    scores = observer.payload("scored")["scores"]
    assert set(scores) == set(config.ARM_NAMES)
    for info in scores.values():
        assert {
            "p",
            "e",
            "ucb",
            "cost_estimate",
            "eligible",
            "reason",
            "selection_score",
        } == set(info)
    assert observer.payload("scored")["chosen"] in config.ARM_NAMES
    store.close()


def test_an_unverifiable_turn_emits_no_update_and_ends_unverified(tmp_path):
    observer = _RecordingObserver()
    # make_router's default verify_fn returns None -- the "cannot tell" path.
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    router.complete("conv", "q", task_type="prose")
    assert "updated" not in observer.names()
    assert observer.payload("verified")["verdict"] is None
    assert observer.payload("verified")["learn"] is False
    assert observer.payload("turn_end")["status"] == "unverified"
    store.close()


def test_a_cascade_emits_one_scored_per_attempt(tmp_path):
    observer = _RecordingObserver()
    # A verifier that always returns 0 escalates every attempt, so the cascade
    # runs to MAX_CASCADE (3) and exhausts.
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 0,
        tmp_path=tmp_path,
        observer=observer,
    )
    router.complete("conv", "q", task_type="code")
    depths = [p["depth"] for name, p in observer.events if name == "scored"]
    assert depths == [0, 1, 2]
    assert observer.payload("turn_end")["status"] == "exhausted"
    assert observer.payload("turn_end")["cascade_depth"] == 3
    store.close()


def test_updated_uses_the_vector_the_arm_was_selected_with(tmp_path):
    # The router's core attribution rule (DESIGN.md section 6): complete()
    # rebuilds x per attempt, so attempt 2's vector has the failure flag set and
    # retry_depth incremented. Reporting attempt 2's vector for attempt 1's arm
    # would misstate the update. Each update's before-values must equal what
    # that same arm scored at its own selection moment.
    observer = _RecordingObserver()
    router, bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 0,
        tmp_path=tmp_path,
        observer=observer,
    )
    router.complete("conv", "q", task_type="code")
    scored = [p for name, p in observer.events if name == "scored"]
    updated = [p for name, p in observer.events if name == "updated"]
    assert len(updated) >= 2, "need at least two cascade steps to distinguish"
    for step in updated:
        matching = [s for s in scored if s["depth"] == step["depth"]]
        assert len(matching) == 1
        chosen = matching[0]["chosen"]
        assert step["arm"] == chosen
        assert step["before"]["ucb"] == pytest.approx(
            matching[0]["scores"][chosen]["ucb"]
        )
        assert step["before"]["p"] == pytest.approx(
            matching[0]["scores"][chosen]["p"]
        )
    # And cross-check against RecordingBandit, which logs the real vectors:
    # the emitted arms must match the arms actually updated, in order.
    assert [s["arm"] for s in updated] == [u[0] for u in bandit.updates]
    store.close()


def test_an_infrastructure_error_emits_called_with_the_error_and_does_not_update(
    tmp_path,
):
    observer = _RecordingObserver()

    def exploding_converse(client, model_id, prompt, **kwargs):
        raise RuntimeError("bedrock is down")

    router, _bandit, store = make_router(
        exploding_converse, tmp_path=tmp_path, observer=observer
    )
    router.complete("conv", "q", task_type="code")
    assert observer.payload("called")["error"] is not None
    assert "updated" not in observer.names()
    # verdict is 0 (escalate) but learn is False (environmental fault).
    assert observer.payload("verified")["verdict"] == 0
    assert observer.payload("verified")["learn"] is False
    store.close()


def test_turn_end_fires_when_no_arm_is_affordable(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    # A budget below every arm's estimate prunes all of them, so select()
    # returns None on the first attempt -- verified: every arm's reason is
    # "over budget" at 1e-9.
    router.complete("conv", "q", task_type="code", budget=1e-9)
    assert observer.payload("turn_end")["status"] == "budget_exhausted"
    assert observer.payload("turn_end")["cascade_depth"] == 0
    assert "called" not in observer.names()
    scores = observer.payload("scored")["scores"]
    assert all(not s["eligible"] for s in scores.values())
    store.close()


def test_record_reward_emits_delayed_update_when_applied(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    result = router.complete("conv", "q", task_type="prose")
    decision_id = result["decision_ids"][0]
    assert router.record_reward(decision_id, 1) is True
    payload = observer.payload("delayed_update")
    assert payload["applied"] is True
    assert payload["decision_id"] == decision_id
    assert payload["pulls_after"] == payload["pulls_before"] + 1
    assert payload["reward"] == 1
    store.close()


def test_record_reward_emits_delayed_update_with_applied_false_on_a_repeat(
    tmp_path,
):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    result = router.complete("conv", "q", task_type="prose")
    decision_id = result["decision_ids"][0]
    router.record_reward(decision_id, 1)
    assert router.record_reward(decision_id, 1) is False
    delayed = [p for name, p in observer.events if name == "delayed_update"]
    assert [p["applied"] for p in delayed] == [True, False]
    assert delayed[1]["pulls_before"] == delayed[1]["pulls_after"]
    assert delayed[1]["reason"] == "already_rewarded"
    store.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `router/.venv/bin/pytest router/tests/test_policy.py -q -k "emits or scored or updated or turn_end"`
Expected: FAIL — `AssertionError: no turn_start event in []`

- [ ] **Step 3: Write the implementation**

In `router/policy.py`, add this helper method (used for both `updated` and `delayed_update`):

```python
    def _arm_state(self, arm, x):
        """This arm's p/e/ucb on the vector it was selected with.

        Not the turn's final vector: complete() rebuilds x per attempt, so
        attempt 2's differs from attempt 1's (DESIGN.md section 6). Costs about
        0.3ms for all seven arms and only runs when an observer is attached.
        """
        info = self.bandit.score(x)[arm]
        return {"p": info["p"], "e": info["e"], "ucb": info["ucb"]}
```

Then wire the call sites in `complete()`:

1. **`turn_id`** — immediately after `remaining` is computed, before the cascade loop:

```python
        turn_id = None
        if self.observer is not None:
            try:
                turn_id = self.observer.next_turn_id()
            except Exception:  # noqa: BLE001 - an observer must not break a turn
                turn_id = None
        self._emit(
            "turn_start",
            turn_id=turn_id,
            conversation_id=conversation_id,
            query=query,
            task_type=task_type,
            budget=total_budget,
            remaining=remaining,
        )
```

2. **`scored`** — immediately after `arm, _scores = self.bandit.select(...)`. Rename `_scores`
   to `scores` at that line and emit:

```python
            self._emit(
                "scored",
                turn_id=turn_id,
                depth=depth,
                chosen=arm,
                remaining=remaining,
                scores=scores,
            )
```

3. **`turn_end` on the no-arm exit** — inside `if arm is None:`, after `status` is assigned and
   before its `return`:

```python
                self._emit(
                    "turn_end",
                    turn_id=turn_id,
                    status=status,
                    cascade_depth=depth,
                    cost=turn_cost,
                    decision_ids=decision_ids,
                )
```

4. **`called`** — wrap the existing `converse_fn` call to time it. Replace the existing
   `try/except` around it with:

```python
            started = time.perf_counter()
            try:
                text, usage = self.converse_fn(
                    self.client, arm, prompt, temperature=0.0
                )
                error = None
            except Exception as exc:  # noqa: BLE001 - recorded as a failure
                text, usage, error = "", {}, str(exc)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._emit(
                "called",
                turn_id=turn_id,
                depth=depth,
                arm=arm,
                text=text,
                cost=None,
                usage=usage,
                error=error,
                latency_ms=latency_ms,
            )
```

   Add `import time` at the top of the module.

   **Corrected after the final review — the original text here was wrong.** It said `cost` is
   `None` because `token_cost` has not run yet, and that "the real cost reaches the UI on
   `turn_end` and via the `updated`/`verified` sequence." I checked both payloads: neither
   carries `cost`. Only `turn_end`'s turn-level aggregate ever arrived, so a cascade gave no
   way to see which attempt spent what, and the page's per-attempt cost line was dead code.
   `token_cost` is a pure function of `(arm, usage)` — a dict lookup and arithmetic, no side
   effects (`arms.py:72-80`) — so it is computed just *above* the emit and the event carries
   the real value. The cost/spend lines below it are otherwise untouched, and their reviewed
   comment about error-path cost samples still stands. Verified: turn cost, status, cascade
   depth, every `decisions` row, `get_spent`, and the bandit update list are byte-identical
   with and without an observer, across the success, cascade, and infrastructure-error paths.

5. **`verified`** — immediately after the `if error: ... else: ...` block that sets `verdict`
   and `learn`:

```python
            self._emit(
                "verified",
                turn_id=turn_id,
                depth=depth,
                arm=arm,
                verdict=verdict,
                learn=learn,
                task_type=task_type,
            )
```

6. **`updated`** — replace the existing `if learn:` block with:

```python
            if learn:
                if self.observer is not None:
                    before = self._arm_state(arm, x)
                    pulls_before = self.bandit.models[arm]["pulls"]
                self.bandit.update(arm, x, float(verdict))
                if self.observer is not None:
                    self._emit(
                        "updated",
                        turn_id=turn_id,
                        depth=depth,
                        arm=arm,
                        reward=float(verdict),
                        pulls_before=pulls_before,
                        pulls_after=self.bandit.models[arm]["pulls"],
                        before=before,
                        after=self._arm_state(arm, x),
                    )
```

   The `if self.observer is not None` guards keep the extra `score()` calls out of the
   no-observer path, which Task 2's equivalence test requires.

7. **`turn_end` on the remaining three returns** — add the same emit before each of the
   `unverified`, `success`, `budget_exhausted` (mid-cascade), and final `exhausted` returns,
   with that path's `status` and `cascade_depth` value. All five `return` statements in
   `complete()` must be covered.

8. **`delayed_update`** in `record_reward` — the method currently has three early `return False`
   guards and one success path. Emit on each. Example for the success path:

```python
        before = None
        pulls_before = None
        if self.observer is not None:
            before = self._arm_state(row["arm"], row["context_vector"])
            pulls_before = self.bandit.models[row["arm"]]["pulls"]
        self.bandit.update(row["arm"], row["context_vector"], float(reward))
        self.store.attach_reward(decision_id, int(reward))
        self._emit(
            "delayed_update",
            decision_id=decision_id,
            arm=row["arm"],
            reward=float(reward),
            applied=True,
            pulls_before=pulls_before,
            pulls_after=self.bandit.models[row["arm"]]["pulls"],
            before=before,
            after=self._arm_state(row["arm"], row["context_vector"]),
        )
        return True
```

   For each `return False` guard, emit `applied=False` with `pulls_before == pulls_after`
   (read once from `self.bandit.models[row["arm"]]["pulls"]`), `before` and `after` both
   `None`, and add a `reason` key naming the guard: `"already_rewarded"`,
   `"not_awaiting_judge"`, or `"projection_mismatch"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `router/.venv/bin/pytest router/tests/test_policy.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm the no-observer path is still identical**

Run: `router/.venv/bin/pytest router/tests -q`
Expected: all pre-existing tests still pass, plus the new event tests. Every pre-existing
`policy` test must pass **unmodified** — if one needed editing, the hook is not inert and that is
a defect, not a test to update. (The only sanctioned exception in this whole plan is
`test_there_is_no_streaming_endpoint`, in Task 4.)

Run: `git status --short router/`
Expected: **no output.** Do not commit.

---

### Task 4: Wire the routes in `app.py`

**Files:**
- Modify: `router/app.py`
- Test: `router/tests/test_viz.py` (append)

**Interfaces:**
- Consumes: `viz.VizObserver`, `viz.sse_stream` (Task 1); `Router(observer=...)` (Task 2).
- Produces:
  - `GET /viz/` → `FileResponse` of `router/static/viz.html`, `media_type="text/html"`
  - `GET /viz/stream` → `StreamingResponse`, `media_type="text/event-stream"`
  - `GET /viz/events?since=N` → `{"events": [...], "stats": {...}}`
  - `build_default_router()` constructs a `VizObserver` and passes it to `Router`
  - `app.state.viz` holds the observer

- [ ] **Step 1: Write the failing tests**

Append to `router/tests/test_viz.py`:

```python
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


def test_viz_events_honours_since():
    client, observer = _app_with_observer()
    observer.emit("a", {})
    observer.emit("b", {})
    with client:
        response = client.get("/viz/events", params={"since": 1})
    assert [e["event"] for e in response.json()["events"]] == ["b"]


def test_viz_stream_sets_the_sse_content_type():
    client, observer = _app_with_observer()
    observer.emit("a", {})
    with client:
        with client.stream("GET", "/viz/stream") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            first = next(response.iter_lines())
    assert first.startswith("data: ")
    assert json.loads(first[len("data: ") :])["event"] == "a"
```

The `/viz/stream` test relies on the replayed ring-buffer event arriving immediately, which is
why it emits before connecting. `sse_stream` yields history first, so the generator does not
block on an empty queue before the first frame.

**Testing `/viz/stream` over HTTP has a hard constraint.** Starlette 1.3.1's `TestClient`
transport is fully buffering: it runs the app to completion into an `io.BytesIO`
(`starlette/testclient.py:353`) and only then builds the `httpx.Response` (line 369). So
`client.stream()` + `iter_lines()` on an endless generator hangs forever — the headers never
arrive, and reading "just the first frame" is impossible too. The test must therefore let the
stream **end**: a timer thread emits a live event and then calls `observer.unsubscribe(...)`,
which is exactly the condition `sse_stream` returns on. The buffered response then completes and
can be asserted on — status, all three headers, and `["replayed", "live"]` in the body, proving
both ring replay and live fan-out went over real HTTP. True incremental streaming is confirmed
under live uvicorn in Task 5 Step 2, not here.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `router/.venv/bin/pytest router/tests/test_viz.py -q -k viz_`
Expected: FAIL — 404 on `/viz/`.

- [ ] **Step 3: Write the implementation**

In `router/app.py`:

Add to the imports:

```python
import pathlib

from fastapi.responses import FileResponse, StreamingResponse

from router import config, judge, viz
```

Add a module-level constant next to the other paths:

```python
_STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
```

In `build_default_router()`, construct the observer and pass it, returning both:

```python
    observer = viz.VizObserver()
    return (
        Router(
            bandit=bandit,
            context_builder=ContextBuilder(),
            cost_estimator=cost_estimator,
            store=Store(config.DB_PATH),
            client=arms.make_client(),
            observer=observer,
        ),
        observer,
    )
```

In `build_app`, handle both shapes so an injected test router still works:

```python
    if router is not None:
        app.state.router = router
        app.state.viz = getattr(router, "observer", None) or viz.VizObserver()
    else:
        app.state.router, app.state.viz = build_default_router()
```

Add the three routes:

```python
    @app.get("/viz/")
    def viz_index():
        """The visualization page. Operator surface, no auth -- same as
        /admin/judge (DESIGN.md section 9)."""
        return FileResponse(_STATIC_DIR / "viz.html", media_type="text/html")

    @app.get("/viz/events")
    def viz_events(since: int = 0):
        observer = app.state.viz
        return {"events": observer.recent(since), "stats": observer.stats()}

    @app.get("/viz/stream")
    def viz_stream(since: int = 0):
        return StreamingResponse(
            viz.sse_stream(app.state.viz, since=since),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
```

The route manages no subscription: `sse_stream` owns that lifecycle, because only the generator's
own `finally` survives disconnect (see Task 1's `sse_stream` note). Stays `def`, not `async def` —
it only constructs a response.

`build_default_router` now returns a tuple. Search for every other caller
(`grep -rn "build_default_router" router/`) and update each. Verified: the only caller today is
`router/app.py:144`.

- [ ] **Step 3b: Narrow the one pre-existing test that `/viz/stream` breaks**

This is the single sanctioned modification of an existing test in this plan, and it needs a
reason on the record.

`router/tests/test_app.py:154` asserts **no route contains the substring `"stream"`**:

```python
def test_there_is_no_streaming_endpoint(client):
    """Streaming and silent escalation are incompatible (DESIGN.md section 5)."""
    paths = set(build_app(router=StubRouter(), lock=False).openapi()["paths"])
    assert not any("stream" in path for path in paths)
```

Adding `GET /viz/stream` fails it. The invariant it protects is that **`/v1/complete` must not
stream model output** — a caller receiving tokens as they arrive cannot have a wrong answer
silently retracted and replaced by an escalated arm. `/viz/stream` carries observer events, never
model tokens, and no client consumes it as an answer, so the invariant is intact. Narrow the
assertion to the surface the invariant is about, and keep the reason in the test:

```python
def test_there_is_no_streaming_endpoint(client):
    """Streaming and silent escalation are incompatible (DESIGN.md section 5).

    Scoped to the /v1 API surface: /viz/stream is an observer event stream, not
    model output, so it cannot expose a retracted answer to a caller.
    """
    paths = set(build_app(router=StubRouter(), lock=False).openapi()["paths"])
    api_paths = {p for p in paths if p.startswith("/v1/")}
    assert api_paths, "guard against the filter silently matching nothing"
    assert not any("stream" in path for path in api_paths)
```

The `assert api_paths` line matters: without it, a future rename of the `/v1/` prefix would empty
the set and the test would pass while checking nothing.

- [ ] **Step 4: Run the tests to verify they pass**

`router/static/` does not exist yet, so create it and a placeholder — Task 5 replaces the file:

```bash
mkdir -p router/static
printf '<html><body>placeholder</body></html>\n' > router/static/viz.html
```

Run: `router/.venv/bin/pytest router/tests/test_viz.py -q`
Expected: PASS, including `test_viz_index_serves_the_html` against the placeholder.

- [ ] **Step 5: Confirm nothing regressed and nothing is staged**

Run: `router/.venv/bin/pytest router/tests -q`
Expected: all pre-existing tests pass (with `test_there_is_no_streaming_endpoint` now narrowed),
plus every `test_viz.py` test.

Run: `git status --short router/`
Expected: **no output.** Do not commit.

---

### Task 5: The UI — `static/viz.html`

**Files:**
- Create: `router/static/viz.html`
- Modify: `router/README.md`

**Interfaces:**
- Consumes: `GET /viz/stream`, `GET /viz/events`, `POST /v1/complete`, and the seven event
  payloads from Task 3.
- Produces: nothing other code depends on.

- [ ] **Step 1: Write the page**

Create `router/static/viz.html`: one self-contained file, no CDN, no build step, no npm.

Required behavior:

1. **Submit form** — `conversation_id` (default `viz-1`), `query`, `task_type` select
   (`math`/`code`/`structured`/`prose`, default `math`), `expected`, `budget` (default `0.02`).
   POSTs JSON to `/v1/complete`.
2. **A visible warning above the button:** "Each submission calls AWS Bedrock and spends real
   money (fractions of a cent)."
3. **Default query** `What is 17 * 23? Put your final answer in \boxed{}.` with `expected`
   `391`. `math` with `\boxed{}` is the path that verifies immediately, so an `updated` card
   appears without waiting on the 10%-sampled judge.
4. **Open the SSE stream before POSTing**, via `new EventSource('/viz/stream')`. Track the
   highest `seq` seen; on `onerror`, reconnect with `?since=<highest_seq>`.
5. **Bucket events by `payload.turn_id`.** SSE is one global stream; ignore turns the page did
   not submit.
6. **Render, in order:** query node → per-attempt block → verdict node → update card, appending
   each new attempt below the previous rather than replacing it.
7. **Each attempt block shows a card per arm, all seven**, sorted by `selection_score`
   descending (nulls last). Each card shows the arm name, `p` and `e` as **two separate bars**,
   `ucb`, `cost_estimate`, and `selection_score`.
   - Mark the `chosen` arm distinctly, and label its `selection_score` as what won — that is
     what `select()` maximizes, not `ucb`.
   - Render `eligible: false` arms greyed, showing their `reason` text.
   - `selection_score` arrives as `null` for ineligible arms (`json_safe` converts `-inf`);
     display `—`, never `null` or `NaN`.
8. **Verdict node:** `1` → "verified ✓", `0` → "wrong → escalating", `null` → "cannot tell →
   deferred to judge". The three must be visually distinct: collapsing `null` into `0`
   misrepresents the router's three-valued contract (VIZ_DESIGN.md §5).
9. **Update card:** `pulls_before → pulls_after`, and `p`, `e`, `ucb` each as `before → after`.
   When no `updated` event arrives for an attempt, show "no bandit update" with the reason from
   `verified.learn`.
10. **`delayed_update` cards** attach to their turn if still rendered, else render standalone in
    a "Delayed rewards (judge)" section.
11. **A "Run judge batch" button** POSTing to `/admin/judge`, showing the returned counts. This
    is how `prose` turns get their reward, so it belongs on the page.
12. **Use `textContent`, never `innerHTML`, for anything derived from an event** — model output
    and error strings reach the DOM, and a model that emits markup would otherwise inject it.
13. **A status line** showing connection state and `stats.dropped`, so a dropped event is
    visible rather than a silent gap.

- [ ] **Step 2: Verify the page loads and the graph renders**

Run: `router/.venv/bin/pytest router/tests/test_viz.py -q`
Expected: PASS, including `test_viz_index_serves_the_html`.

Then, manually — this is the only step in the plan that spends money:

```bash
router/.venv/bin/uvicorn router.app:build_app --factory --workers 1 --port 8000
```

Open `http://localhost:8000/viz/`, submit the default query, and confirm:
- seven arm cards appear, the chosen one marked
- the verdict node reads "verified ✓"
- the update card shows `pulls` incrementing and `p`/`e`/`ucb` before → after
- switching `task_type` to `prose` yields "cannot tell → deferred to judge" and no update card
- setting `budget` to `0.0000001` greys every arm with "over budget" and shows `—`, and the
  browser console reports **no** `JSON.parse` error

- [ ] **Step 3: Document it**

`router/README.md:60` currently reads:

> Responses are buffered; there is no streaming endpoint. Streaming and silent
> escalation are incompatible (DESIGN.md §5).

`/viz/stream` makes that flatly false as written, so correct it in place rather than leaving the
contradiction:

```markdown
Answers are buffered; `/v1/complete` never streams. Streaming an answer and
silently escalating it are incompatible (DESIGN.md §5). `GET /viz/stream` is a
separate observer feed — routing events, never model output.
```

Then add the three `/viz` routes to the endpoint list:

```markdown
- `GET /viz/` — live routing visualization (see below)
- `GET /viz/stream` — SSE feed of routing events
- `GET /viz/events?since=N` — recent events as JSON, for catch-up after a reconnect
```

and a short section:

```markdown
## Live visualization

    http://localhost:8000/viz/

Shows one turn's routing decision: per-attempt arm scores, the verdict, and the LinUCB
before/after for the arm that was updated. Submitting from this page calls Bedrock and
spends real money. Events are in-memory only, so a restart clears the view; SQLite remains
the durable record.
```

- [ ] **Step 4: Confirm the whole suite passes and nothing is staged**

Run: `router/.venv/bin/pytest router/tests -q`
Expected: the full suite green — all pre-existing tests plus all new ones. Report the actual
numbers rather than matching a count in this plan.

Run: `git status --short router/`
Expected: **no output.** Do not commit.

---

## Notes for the implementer

- **Expected test counts are approximate.** The plan's totals assume every listed test lands as
  written. If your count differs but nothing pre-existing broke, that is fine — report the
  number. `170 passed, 3 skipped` before this plan is the number that must never regress.
- **`policy.py` is the most heavily reviewed file in this project** — six review findings and
  two re-review rounds. Keep every change additive and guarded. Do not refactor the
  cost-accounting or `logged_reward` lines; their comments record decisions that were argued
  through.
- **Exactly one pre-existing test may be modified:** `test_there_is_no_streaming_endpoint`, per
  Task 4 Step 3b, which explains why. Beyond `make_router` gaining an `observer=None` keyword,
  **no other existing test may change.** The hook is meant to be inert; any other test that must
  change is evidence it is not — stop and report rather than editing the test.
- **Do not trust the counts, run the suite.** `170 passed, 3 skipped` is the pre-plan baseline
  (with no dev server running). After each task, what matters is that no previously-passing test
  is failing — not that the total matches a number in this document.
