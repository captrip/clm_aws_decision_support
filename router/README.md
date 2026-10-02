# Per-turn LLM router

Local prototype. Not committed — `router/` is in `.git/info/exclude`.

See [DESIGN.md](DESIGN.md) for the design and [PLAN.md](PLAN.md) for the build steps.

## Setup

    /opt/anaconda3/bin/python3.13 -m venv router/.venv
    router/.venv/bin/pip install -r router/requirements.txt

## Tests

    router/.venv/bin/pytest router/tests -v          # all mocked, no AWS
    ROUTER_SMOKE=1 router/.venv/bin/pytest router/tests/test_smoke.py -v

## Offline replay

    router/.venv/bin/python -m router.replay

Scores the router against the 500 logged cascade runs in `Result/algo1res.json`
with zero spend. Read it as a wiring-and-cost sanity check, not an accuracy
estimate, for two reasons:

- That log is the **OpenRouter** experiment. None of its model names is one of
  this router's seven Bedrock arms, so `replay._ARM_ALIASES` maps each logged
  model onto the arm nearest it in capability tier. Correctness and cost
  therefore come from a different model than the arm being credited. Unmapped,
  the name sets are disjoint and all 500 records score as unknown.
- Each record logs outcomes for only the ~1.25 arms the original cascade
  actually tried. Any step landing on an unlogged arm counts as `unknown`, never
  as wrong — so `unknown_outcomes` stays high by construction. Only the 6 arms
  the log can score are offered as candidates; an arm with no outcome anywhere
  would otherwise keep its full initial optimism and absorb nearly every pick.

`Algorithm/M91_AWS.json` is the only log whose names are genuinely these arms,
and it holds 5 records — too few to measure anything.

## Run the service

**Single worker only.** Two workers each keep their own bandit state and
overwrite each other's snapshot; the service keeps serving and silently stops
learning. `app.py` holds an exclusive lock to make that impossible.

    router/.venv/bin/uvicorn router.app:build_app --factory --workers 1 --port 8000

Requires AWS profile `258574424891` with Bedrock access and model access granted
for all seven arms plus `us.anthropic.claude-sonnet-5`.

## Endpoints

- `POST /v1/complete` — `{conversation_id, query, task_type, expected?, schema?, turn_index?, conversation_tokens?, budget?}`
- `POST /v1/reward` — `{decision_id, reward}` for delayed feedback. Only a
  decision awaiting a verdict is eligible; one already scored, or one whose arm
  raised an infrastructure error, reports `applied: false`.
- `POST /admin/judge` — run one sampled-judge batch; returns `{considered, judged, skipped, ambiguous, errors}`. Operator-triggered; there is no scheduler.
- `GET /healthz` — arms and pull counts
- `GET /viz/` — live routing visualization (see below)
- `GET /viz/stream` — SSE feed of routing events
- `GET /viz/events?since=N` — recent events as JSON, for catch-up after a reconnect
- `GET /viz/stats` — just the counters (`emitted`, `dropped`, `subscribers`); the page
  polls this rather than downloading the whole ring to read three integers

Answers are buffered; `/v1/complete` never streams. Streaming an answer and
silently escalating it are incompatible (DESIGN.md §5). `GET /viz/stream` is a
separate observer feed — routing events, never model output.

## Live visualization

    http://localhost:8000/viz/

Shows one turn's routing decision: per-attempt arm scores, the verdict, and the LinUCB
before/after for the arm that was updated. Submitting from this page calls Bedrock and
spends real money. Events are in-memory only, so a restart clears the view; SQLite remains
the durable record.

`p` and `e` are drawn as separate bars diverging from a centre baseline, because `p`
(the mean, `x·θ`) is signed and routinely negative, while `e` (the confidence width) is
not. Bar length is scaled by the largest absolute value among the seven arms in that
same attempt, so bars are comparable within an attempt but not across attempts — the
number beside each bar is the truth, the bar is the glance.

The status line reports drops twice: **Dropped here** counts events *this tab* never
received, inferred from gaps in `seq`; **Dropped (all tabs)** is the process-wide count.
Subscriber queues bound at 256 and drop rather than block, so a slow tab loses events by
design instead of stalling a live turn.

## Known gaps

Cold start, prompt-cache affinity, multi-worker state, and auth are all out of
scope — see DESIGN.md §9.

The judge's own Sonnet spend is not counted by any metric the service reports:
`config.JUDGE_MODEL` is not in `config.ARMS`, so `arms.token_cost` would raise
`KeyError` on it. Judge cost is real and invisible here.

A conversation can overshoot its budget by one call. The affordability gate is a
*lower* confidence bound by design, so it is systematically optimistic before a
call and correct after it.
