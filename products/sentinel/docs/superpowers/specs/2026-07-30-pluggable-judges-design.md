# Pluggable Judges — Design

**Date:** 2026-07-30
**Status:** Approved, not yet implemented

## Problem

The Governor's sentinel capability has exactly one extension point today: `Governor(ref, judge=...)`
accepts any object satisfying the `TrajectoryJudge` protocol (`sentinal/core/trajectory.py:34`).
That is enough to swap the judgment policy wholesale, but not enough to build on:

1. **One judge per Governor.** A user who wants the built-in loop detector *and* a business
   metric must fork `HeuristicJudge` or hand-roll a compositing judge.
2. **Judges see only `list[TrajectoryStep]`.** Loop detection works on that. A latency budget
   or a spend cap does not — wall-clock and session memory are unreachable.
3. **No structured channel for business detail.** `Correction` carries a free-text `note`.
   A judge that trips on `spend=2.14` can only stringify it, so the number is unavailable to
   the dashboard, the interventions log, or any downstream consumer.
4. **Authoring a judge means writing a class.** Reasonable for `HeuristicJudge`, which holds
   thresholds. Heavy for a cost cap, which is a predicate.
5. **Wiring is order-sensitive and silent.** A Governor must be `register()`ed *before*
   `runtime.execute()`, or the runtime lazily builds a default one
   (`sentinal/runtime/runtime.py:170`) and the user's judge is dropped with no warning.

## Goals

- Users are not bound by the default judges.
- A curated built-in set covers the failure modes an agentic loop actually has.
- Onboarding a new business metric/judge is a small, obvious amount of code.
- No breaking changes: every existing judge, test, and example keeps working.

## Non-goals

- Extending the `CorrectionFlag` vocabulary. Deliberately closed — see Decisions.
- Extracting the actor-side correction ladder. Noted as future work below.
- A built-in cost/spend judge. See Decisions.

## Decisions

**Closed flag vocabulary, structured metadata.** Business judges map their finding onto the
existing five verbs and carry the detail in `note` + a new `meta` dict. The alternative — a user-
extensible flag registry with user-supplied actor-side handlers — was rejected because every new
flag would require handling code in every executable, and an unhandled flag becomes a
silent-failure class the framework would have to define behavior for. Keeping the verbs closed is
what lets the actor-side ladder (`examples/search_agent.py:251-262`) stay universal and unchanged.

**Severity arbitration, all findings recorded.** With multiple judges, the Governor pushes one
unambiguous order per step but records every finding. "First non-CONTINUE wins" was rejected
because a cheap loop detector registered first would mask a `STOP`-worthy safety hit. "Push all"
was rejected because the actor could drain `REDIRECT` + `STOP` together and the outcome would
depend on drain order.

**Explicit attachment, no global registry.** `@judge` returns an object; it does not
self-register. Auto-registration was rejected: import order would become behavior, tests would
leak judges into each other, and disabling one judge would mean not importing a module.

**No built-in cost judge.** Token/USD usage metadata differs per provider and per streaming mode.
A built-in would either guess or silently read zero and report `$0.00` — worse than absent. Cost
ships as the documented example of a *user-defined* judge instead.

**No built-in `ToolChoiceJudge`.** The LLM-based tool router in `examples/search_agent.py:331`
would put a network call inside judging, on the Governor's polling thread. It stays in `examples/`
as the reference implementation for a bespoke judge.

## Architecture

Three layers, each independently testable:

```
sentinal/core/trajectory.py     types: Correction.meta, JudgeContext, TrajectoryJudge
sentinal/judges/                built-ins + @judge decorator + flag constructors
sentinal/actors/governor.py     runs N judges, arbitrates by severity, records all
sentinal/runtime/runtime.py     threads judges= through to the Governor it builds
```

Data flow per polled step, unchanged in shape from today:

```
executable.append_step()  ->  session trajectory
                                    |
Governor polls (0.1s), builds JudgeContext
                                    |
                     runs every judge, collects verdicts
                                    |
        records all  ->  _interventions      pushes max-severity  ->  session
                                                                          |
                                              executable.drain_corrections()
```

### Layer 1 — Types (`sentinal/core/trajectory.py`)

`Correction` gains one optional field:

```python
@dataclass(frozen=True)
class Correction:
    flag: CorrectionFlag
    note: str | None = None
    step_index: int = -1
    ts: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)   # NEW
```

`meta` is free-form and **never interpreted by the framework**. It is carried into
`get_interventions()` and the dashboard so structured detail (`spend`, `severity`, `metric`)
survives to the operator. `test_judge_is_pure` (equality of two `judge()` results) still holds:
dict equality is by value.

`JudgeContext` is what a judge receives:

```python
@dataclass(frozen=True)
class JudgeContext:
    trajectory: list[TrajectoryStep]
    elapsed_s: float              # wall-clock since govern_session started
    memory: Mapping[str, Any]     # read-only snapshot of session memory
    input: Mapping[str, Any]      # context.input, e.g. {"query": ...}

    def __len__(self): return len(self.trajectory)
    def __getitem__(self, i): return self.trajectory[i]
    def __iter__(self): return iter(self.trajectory)
```

Those three dunders are the backwards-compatibility mechanism. `JudgeContext` is sequence-like,
so a judge written against `list[TrajectoryStep]` works unchanged when handed a context:
`len(trajectory)`, `trajectory[-1]`, and slicing (`trajectory[-3:]`, used at
`sentinal/actors/judges.py:28`) all behave. No shim, no signature inspection, no migration.

`memory` is a read-only mapping so a judge cannot mutate session state; `input` is passed rather
than the session so judges never gain access to `push_correction`, which would bypass arbitration.

The protocol widens:

```python
@runtime_checkable
class TrajectoryJudge(Protocol):
    def judge(self, ctx: JudgeContext) -> Correction | None: ...
```

Returning `None` means "nothing to say", so a decorated function can fall off the end rather than
construct a `CONTINUE`. The Governor already tolerates `None` (`sentinal/actors/governor.py:111`).

Severity rank lives beside the enum as a readable module-level map:

```python
SEVERITY = {CONTINUE: 0, REDIRECT: 1, AVOID_TOOL: 2, FORCE_FINALIZE: 3, STOP: 4}
```

### Layer 2 — Authoring (`sentinal/judges/`)

`@judge` wraps a function into a Protocol-satisfying object that also remains callable as the
original function, so it is directly unit-testable:

```python
@judge(name="cost_cap")
def cost_cap(ctx):
    spend = ctx.memory.get("usd", 0.0)
    if spend > 2.0:
        return finalize(f"cost cap exceeded: ${spend:.2f}", meta={"spend": spend})
```

Four flag constructors — `redirect()`, `avoid_tool()`, `finalize()`, `stop()` — each taking
`note` and `meta`, so a judge body reads as intent rather than enum plumbing.

Attachment is explicit:

```python
runtime.execute(session, judges=[StepBudgetJudge(), cost_cap])
```

### Layer 3 — Governor

```python
def __init__(self, *args, quality_threshold: float = 0.7,
             judge: TrajectoryJudge | None = None,
             judges: Sequence[TrajectoryJudge] | None = None, **kwargs):
    self._judges = list(judges) if judges else ([judge] if judge else [HeuristicJudge()])
```

`judge=` remains supported and means a one-element list. Passing **both** raises `ValueError` —
silently picking one is the kind of ambiguity that costs an afternoon.

Replacing `sentinal/actors/governor.py:104-123`:

```python
verdicts = []
for j in self._judges:
    try:
        c = j.judge(ctx)
    except Exception as ex:          # one broken judge must not kill the run
        logger.warning(f"judge {_judge_name(j)} failed: {ex}")
        continue
    if c is not None and c.flag != CorrectionFlag.CONTINUE:
        verdicts.append((j, c))

for j, c in verdicts:               # every finding is recorded
    self._interventions.append({"type": "correction", "judge": _judge_name(j),
                                "flag": c.flag.value, "note": c.note,
                                "meta": dict(c.meta), "pushed": False})
if verdicts:
    winner_j, winner_c = max(verdicts, key=lambda v: SEVERITY[v[1].flag])
    # mark the winner's recorded entry pushed=True
    session.push_correction(winner_c)
    activity_bus.emit(...)
```

Ties break by registration order (first judge wins) — deterministic, and lets ordering express
preference.

Two properties this buys:

- **Nothing is invisible.** A suppressed `REDIRECT` still appears in `get_interventions()` with
  `pushed: False`, so an operator asking "why didn't my judge fire" can see that it fired and lost.
- **Fault isolation is per-judge.** Today one raising judge loses the whole step's judgment
  (`governor.py:108` sets `correction = None`); now its peers still get their say.

`_interventions` entries gain `judge`, `meta`, and `pushed`. Existing consumers are unaffected:
`examples/search_agent.py:498` filters on `type == "correction"` and reads `flag`/`note`. Its
summary print will be extended to name the judge that spoke.

**Re-judgment trigger.** Today judging runs only when the trajectory *grew*
(`len > _last_judged_len`). That gate matters more with N judges — it prevents re-running an
expensive judge against an unchanged trajectory. But it means a purely time-based judge cannot
fire on a stalled trajectory: if the agent hangs mid-tool-call, no new step arrives and
`LatencyBudgetJudge` is never asked. Fix: re-judge when the trajectory grew **or** ≥1s has passed
since the last judgment. Cheap, and it makes the latency guard meaningful.

Judging remains on the Governor's polling thread at `poll_interval=0.1`.

### Runtime wiring

`SentinelRuntime.execute(session, judges=None)` threads `judges` into the Governor it lazily
builds (`sentinal/runtime/runtime.py:171`). This closes the silent-drop trap: the common path no
longer requires manually constructing and registering a Governor before `execute()`. If a Governor
was pre-registered *and* `judges=` is passed, `judges=` wins and logs that it is overriding —
explicit beats stale.

## Built-in set

The four failure modes an agentic loop actually has:

| Judge | Fires when | Verb | Origin |
|---|---|---|---|
| `StepBudgetJudge(max_steps=20)` | runaway ReAct loop | `FORCE_FINALIZE` | extracted |
| `RepeatToolJudge(threshold=3)` | same tool repeated | `AVOID_TOOL` | extracted |
| `OutputLoopJudge(window=3)` | model repeating itself | `REDIRECT` | extracted |
| `LatencyBudgetJudge(max_seconds)` | wall-clock blown / stall | `FORCE_FINALIZE` | new (~10 lines) |

Three of four are a pure decomposition of `HeuristicJudge` (`sentinal/actors/judges.py:22-39`) —
no new logic, and its five existing tests carry over per-judge. The fourth is trivial once
`elapsed_s` is in `JudgeContext`.

`HeuristicJudge` survives as a thin alias composing the first three, so
`examples/search_agent.py:453` and all current tests are untouched.

## Error handling

| Failure | Behavior |
|---|---|
| A judge raises | Logged with the judge's name, that judge skipped for the step, peers still run |
| A judge returns a non-`Correction` | Treated as no-op, logged once per judge |
| A judge returns `None` | No-op (the normal "nothing to say" path) |
| Both `judge=` and `judges=` passed | `ValueError` at construction |
| No judges given | Defaults to `HeuristicJudge()`, as today |
| Judge mutates `ctx.memory` | Prevented: read-only mapping |

## Testing

- Unit tests per built-in judge, carrying over the five existing `HeuristicJudge` cases.
- `HeuristicJudge` alias produces the same verdicts as the three extracted judges — guards the
  decomposition.
- `JudgeContext` is sequence-like: a list-based judge works unchanged against a context.
- Governor: three judges disagreeing → most severe pushed, all three recorded, winner marked.
- Governor: a raising judge does not suppress its peers.
- Governor: `judge=` and `judges=` both passed → `ValueError`.
- `@judge`-decorated functions and class judges compose in one `judges=` list.
- Latency judge fires on a stalled (non-growing) trajectory — guards the re-judgment trigger.
- Integration: a custom business judge attached via `judges=` steers a real session, mirroring
  `tests/test_trajectory/test_integration.py` with a fake executable and no live LLM.

## Scope

Touches `sentinal/core/trajectory.py`, `sentinal/actors/governor.py`,
`sentinal/runtime/runtime.py`, `sentinal/__init__.py` (exports); adds `sentinal/judges/`.

The actor-side correction ladder in `examples/search_agent.py:251-262` is **unchanged** — that is
the payoff of keeping the five verbs closed. The ladder is duplicated per-executable and would be
worth extracting into a reusable helper, but that is separate work and is not folded in here.
