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

import anyio
import numpy as np

RING_SIZE = 500
QUEUE_SIZE = 256


def json_safe(value, _depth=0):
    """Convert to something json.dumps emits as strict, parseable JSON.

    bandit.select assigns an ineligible arm selection_score = float("-inf"),
    and json.dumps writes that as a bare -Infinity token. That is invalid JSON:
    the browser's JSON.parse throws and the whole event stream dies. Since
    VIZ_DESIGN.md section 5 requires showing ineligible arms, the first
    budget-pruned turn would break the UI. Non-finite floats become null.

    numpy types are handled here too -- json.dumps raises TypeError on
    np.float64, and the score payload is full of them.

    Protects emit() from raising by:
      - capping recursion depth (self-referential or deeply nested structures)
      - defensively calling str() (a __str__ that raises does not propagate)
    """
    if _depth > 32:
        return "<max depth>"
    if isinstance(value, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, _depth + 1) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v, _depth + 1) for v in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item(), _depth + 1)
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str)) or value is None:
        return value
    try:
        return str(value)
    except Exception:
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
        except Exception as exc:
            # Belt-and-braces: json_safe should not raise, but if any exception
            # escapes, emit() must still honor its "never raises" contract.
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

    def subscribe_with_history(self, since=0):
        """Register a queue AND read the replay backlog under ONE lock.

        Doing these as two separate acquisitions leaves a window where an
        emit() lands in both the new queue and the backlog, so the client sees
        that event twice. A duplicate is not harmless: a repeated `scored`
        renders a phantom extra attempt, shifting every later answer by one and
        showing a routing history that never happened.
        """
        q = queue.Queue(maxsize=self.queue_size)
        with self._lock:
            backlog = [e for e in self._ring if e["seq"] > int(since)]
            self._subscribers.append(q)
        return q, backlog

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

    Registers and reads its backlog atomically (subscribe_with_history), so no
    event is delivered both as replay and as live -- a duplicate renders a
    phantom attempt on the page.
    """
    q, backlog = observer.subscribe_with_history(since)
    try:
        for record in backlog:
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
