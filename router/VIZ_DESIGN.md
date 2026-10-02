# Router Visualization — Design

**Status:** approved design, not yet implemented
**Date:** 2026-08-03
**Scope:** local-only. This directory is excluded from git via `.git/info/exclude`.
Not committed, and deliberately not written to `docs/superpowers/specs/` for that reason.

A live view of one turn's routing decision: for each cascade stage, why every arm scored as
it did, which was picked, what the verifier said, and exactly how that arm's LinUCB state
moved. Extends the router in [DESIGN.md](DESIGN.md) without altering its decisions.

---

## 1. Goal

Make the mechanism visible. Given a query submitted from the browser, show:

- the 7 arms' scores at each attempt, with the winner and the reason each loser lost
- the verdict, and whether it produced a bandit update
- for the arm that was updated: `pulls`, `p` (mean), `e` (confidence width), and `ucb`
  before and after, **evaluated on that arm's own context vector**

Explicit non-goals, and each is a deliberate cut:

- **No monitoring dashboard.** No learning curves, cost-over-time charts, or trend lines.
  This view explains one decision; monitoring is a different tool with a different layout.
- **No θ inspector or context-vector heatmap.** 49 numbers per arm is data, not insight.
- **No control over routing.** The UI cannot pin an arm, replay with a different budget, or
  edit bandit state. It observes.

### Why this is mostly surfacing, not instrumenting

`bandit.select()` already returns per-arm `p`, `e`, `ucb`, `cost_estimate`, `eligible`,
`reason`, and `selection_score` for all 7 arms on every attempt. `policy.py` discards it as
`_scores`, and `app.py` drops the `attempts` list from the HTTP response. The only genuinely
new computation is one `bandit.score(x)` after each update, to get the after-values.

Measured: re-scoring all 7 arms costs **0.31 ms**, against a multi-second Bedrock call.

---

## 2. Approach

An **observer hook** on `Router`, an in-process event hub, and one static HTML file.

```
browser  ──POST /v1/complete──────────────────────┐
         ──GET  /viz/stream (SSE, held open)──┐   │
                                              │   ▼
router/static/viz.html          router/viz.py │  policy.Router.complete()
  (vanilla JS + SVG, no build)   VizObserver ─┘    │ emits at 6 points
                                  ring buffer      └─ observer.emit(...)
                                  + subscriber queues
```

The browser opens the SSE stream **before** POSTing, so events arrive while Bedrock calls
are still in flight — attempt 1's outcome renders while attempt 2 is being made.

`viz.py` holds state in process memory, so it inherits the single-worker invariant
(DESIGN.md §3) for free, and adds one more reason it must hold: two workers would each
serve a partial event stream.

### Rejected alternatives

- **Poll SQLite from a separate read-only process.** Requires no change to `policy.py`, but
  `decisions` stores only the chosen arm's vector — never the per-arm scores. Reconstructing
  them would use the *current* `A`/`b` and show what the bandit believes now, not what it
  believed when it chose. For a mechanism explainer that is not a limitation, it is wrong
  output.
- **Return `attempts` + scores inline from `/v1/complete`.** No SSE, no ring buffer, but it
  makes debug payload part of the public API contract, ships ~2 KB to every caller, and
  cannot show anything mid-turn: the whole cascade appears at once, after every Bedrock call
  has completed. That forfeits the live escalation view, which is the point.

The observer keeps the feature entirely removable: delete `viz.py`, pass no observer, and
the router is exactly what it is today.

---

## 3. The observer hook

`Router.__init__` gains one optional parameter, `observer=None`. The whole integration
surface into the cascade is one private method:

```python
def _emit(self, event, **payload):
    """Fire a visualization event. Never let it affect the turn.

    A visualization that can break routing is worse than no visualization: the
    observer is a debugging aid and the turn is the product. Any exception from a
    subscriber -- a full queue, a disconnected browser, a bug in viz.py -- is
    swallowed here.
    """
    if self.observer is None:
        return
    try:
        self.observer.emit(event, payload)
    except Exception:  # noqa: BLE001 - deliberate: see docstring
        pass
```

**With `observer=None` the router behaves identically to today**, down to the absence of any
extra computation. This is the property the existing 170 tests protect and §6 asserts
directly.

### Events

Six from `complete()`, one from `record_reward()`. Every event carries `turn_id`, `seq`,
`event`, and a payload.

| Event | Emitted where | Payload |
| :--- | :--- | :--- |
| `turn_start` | top of `complete()` | `conversation_id`, `query`, `task_type`, `budget`, `remaining` |
| `scored` | after `bandit.select()` | `depth`, `chosen`, and per-arm `p`, `e`, `ucb`, `cost_estimate`, `eligible`, `reason`, `selection_score` for all 7 |
| `called` | after `converse_fn` | `arm`, `text`, `cost`, `usage`, `error`, `latency_ms` |
| `verified` | after `verify_fn` | `verdict` (`1`/`0`/`None`), `learn`, `task_type` |
| `updated` | after `bandit.update()` | `arm`, `reward`, `pulls_before`/`pulls_after`, and `p`/`e`/`ucb` before and after |
| `turn_end` | every `return` | `status`, `cascade_depth`, `cost`, `decision_ids` |
| `delayed_update` | `record_reward()` | `decision_id`, `arm`, `reward`, `applied`, same before/after fields |

`scored` is exactly the `_scores` value `policy.py` currently throws away.

`delayed_update` exists because without it the judge path is invisible. `prose` is the
default task type and never verifies, so its only reward arrives via `POST /admin/judge` or
`POST /v1/reward`. A view of the mechanism that omits the judge omits how most turns learn.

`turn_end` must be emitted on **all five** of `complete()`'s `return` statements. They carry
four distinct statuses, because the no-eligible-arm exit chooses between two of them:

1. no eligible arm → `budget_exhausted` at `depth == 0`, else `exhausted`
2. `verdict is None` → `unverified`
3. `verdict == 1` → `success`
4. budget spent mid-cascade → `budget_exhausted`
5. cascade loop exhausted → `exhausted`

A missing one leaves the UI showing a turn that never finishes.

### The before/after rule

`updated` reports `p`/`e`/`ucb` computed on **the same `x` that arm was selected with**, not
the turn's final vector. `complete()` rebuilds `x` per attempt — attempt 2's vector has the
failure flag set and `retry_depth` incremented — so attributing attempt 2's vector to
attempt 1's arm misreports the update. This is the same per-arm attribution rule the router
is built around (DESIGN.md §6), and the visualization must not be the one place it is
broken. §6 tests it.

Why both `p` and `e`, not just `ucb`: measured on a fresh arm, one successful update moved
the mean `p` from `+0.000` to `+0.816` while the confidence width `e` collapsed from `+1.423`
to `+0.610` — so `ucb` moved only `+1.4230` to `+1.4262`. Showing the sum alone makes
learning look like it is not happening. The two terms are shown separately.

---

## 4. The event hub

`router/viz.py`. One class plus three route handlers wired in `app.py`.

```
VizObserver
  emit(event, payload)          -> stamp turn_id/seq, append to ring, fan out
  subscribe() -> queue.Queue    -> register a bounded queue
  unsubscribe(q)
  recent(since)                 -> ring-buffer replay for late joiners
```

### Two hazards, and the rules that follow

**A slow browser must not stall a real turn.** `complete()` runs on Starlette's threadpool;
a blocking `put` onto a full queue would freeze a live Bedrock turn behind a browser that
stopped reading. Every subscriber queue is therefore **bounded and drops on full** — never
blocks, never grows without limit. The visualization degrades; the router does not.

**Concurrent turns interleave on one stream.** SSE is global. Each `complete()` call takes a
monotonic `turn_id`; the UI buckets events by it and ignores turns it did not submit. `seq`
is monotonic across the whole hub so a client can detect gaps from drops.

A `deque(maxlen=500)` ring buffer backs `GET /viz/events?since=N`, so a browser that
connects late or reconnects catches up instead of rendering blank. A `threading.Lock` guards
the subscriber set and the ring — the same discipline as `store.py:58`, and required for the
same reason: sync handlers share one object across ~40 threads.

### Routes

| Route | Returns |
| :--- | :--- |
| `GET /viz/` | `router/static/viz.html` |
| `GET /viz/stream` | SSE, `text/event-stream`, one JSON event per frame |
| `GET /viz/events?since=N` | ring-buffer replay as JSON |

`/viz/*` is an operator surface like `/admin/judge`: no auth, consistent with DESIGN.md §9
placing auth out of scope. `viz.html` echoes model output into the DOM, so it must set text
content rather than `innerHTML` — a model that emits markup would otherwise inject it.

---

## 5. The UI

One self-contained `router/static/viz.html`: vanilla JS, hand-drawn SVG/CSS, no npm, no
build step, no CDN. It renders top to bottom as events arrive:

```
┌───────────────────────────────────────────┐
│ QUERY  "17 * 23, box your answer"  math   │
└─────────────────────┬─────────────────────┘
                      ▼
  ATTEMPT 1   depth=0   budget $0.0200
 ┌───────────────┬───────────────┬─────────┐
 │★nova-micro    │ nova-lite     │ mistral │
 │ p ▁ e ████    │ p ▁ e ████    │ p ▁ e ██│
 │ ucb   +0.212  │ ucb   +0.212  │  +0.212 │
 │ cost $0.00009 │ cost $0.00015 │ $0.00010│
 │ score +0.098 ▲│ score +0.087  │  +0.081 │
 └───────────────┴───────────────┴─────────┘
         │ picked ★  → "\boxed{391}"
         ▼
   VERDICT ✅ 1  (verify.check, math)
         ▼
 ┌───────────────────────────────────────────┐
 │ BANDIT UPDATE  nova-micro                 │
 │ A += xxᵀ   b += 1.0·x   pulls 3 → 4       │
 │ p  +0.000 → +0.816    e  +1.423 → +0.610  │
 │ ucb +1.4230 → +1.4262                     │
 └───────────────────────────────────────────┘
```

Rendering rules that carry meaning:

- **`p` and `e` are separate bars** on every arm card, per §3.
- **The winner is marked**, and its `selection_score` — not its `ucb` — is what won, since
  that is what `select()` maximizes.
- **Ineligible arms render greyed with their `reason`** ("over budget", "not a candidate"),
  never omitted. Budget pruning and already-tried arms are part of the decision and must be
  visible as such.
- **A cascade appends a new attempt row** rather than replacing the previous one, so the
  escalation is readable as a sequence.
- **`verdict: None` renders as "cannot tell → deferred to judge"**, visibly distinct from
  `0`. Collapsing the two would misrepresent the router's core three-valued contract.
- **A `delayed_update` card can arrive minutes later** and attaches to its original turn if
  still on screen, otherwise renders standalone.

An in-page **Submit** form POSTs to `/v1/complete`. It carries a plain warning that each
submission spends real money, and defaults `task_type` to `math` with a `\boxed{}` hint,
because that is the path that verifies immediately and therefore shows an `updated` card
without waiting on the judge.

---

## 6. Testing

All mocked, no AWS, no browser driver. New `router/tests/test_viz.py`, plus additions to
`test_policy.py`.

**The router must not change:**

- `observer=None` emits nothing, and every existing `complete()` assertion still holds
- **an observer whose `emit` raises on every event does not break the turn** — the single
  most important test here; a visualization that can take down routing is a net loss

**Events are correct:**

- sequence per path: verified success (1 attempt), `prose`/unverified, a 3-arm cascade, an
  infrastructure error
- `turn_end` fires on all five `return` statements, covering all four statuses
- `scored` carries all 7 arms, including ineligible ones with their `reason`
- `updated`'s before/after are computed on the vector that arm was **selected** with, not
  the turn's last vector — asserted by driving a 2-attempt cascade with distinguishable
  vectors and checking arm 1's card against arm 1's `x`
- `delayed_update` fires from `record_reward`, and reports `applied: false` for the
  already-rewarded, wrong-projection, and infra-error rows

**The hub is safe:**

- the ring buffer stays at its bound under more events than it can hold
- a **full subscriber queue drops rather than blocking** — asserted with a queue of size 1
  and a timed `emit`
- `recent(since)` returns only newer events; an out-of-range `since` is handled
- `/viz/stream` yields well-formed `data: {json}\n\n` frames; `/viz/` serves the HTML

---

## 7. Files

| | |
| :--- | :--- |
| Create | `router/viz.py`, `router/static/viz.html`, `router/tests/test_viz.py` |
| Modify | `router/policy.py` — `observer` param, `_emit`, 7 call sites |
| Modify | `router/app.py` — construct `VizObserver`, wire it, add 3 `/viz` routes |
| Spec | this file |

`config.py` is untouched: no new constants are needed that are not local to `viz.py`.

Nothing under `router/` is ever staged. Verified with `git status --short router/`, which
must print nothing.

---

## 8. Accepted limitations

- **`policy.py` is the most heavily reviewed file in the router** — six findings and two
  re-review rounds. This edit is additive and guarded, but it is a real change to
  load-bearing code, and the `observer=None` equivalence test exists because of that.
- **With ~10 recorded pulls the arm cards will look nearly identical.** The UI will show the
  machinery clearly and learning faintly. Dramatic divergence needs hundreds of rewarded
  turns; `math` turns with `\boxed{}` and correct `expected` are the fastest way there,
  since they verify at 100% instead of the judge's 10% sample.
- **Events are memory-only.** A restart loses the stream history; SQLite remains the durable
  record. Persisting events was rejected as monitoring, which §1 cuts.
- **No auth on `/viz/*`**, consistent with `/admin/judge` and DESIGN.md §9.
- **Judge spend is still uncounted** (README.md "Known gaps"). A `delayed_update` card shows
  that a judge verdict landed, not what the judge cost.
