# User-selectable priority: multi-objective selection

Local prototype design. Not committed — `router/` is in `.git/info/exclude`.

Extends [DESIGN.md](DESIGN.md). Read §4 (context vector), §7 (cost estimation)
and [bandit.py](bandit.py) `select` first.

## 1. Goal

Let the caller say what they care about — quality, cost, latency, brevity,
format compliance — and have the router act on it, instead of hardcoding one
fixed cost/quality trade-off.

    POST /v1/complete  {..., "priority": "cheap"}

## 2. What exists today, precisely

The bandit learns exactly **one** thing: `P(correct | x)` per arm, via
`A += xxᵀ`, `b += r·x`. That is the only learned quantity in the system.

Cost is *not* learned by the bandit. It enters selection through two separate,
hardcoded mechanisms in `bandit.select`:

1. **Hard pruning** — `cost_estimate > remaining_budget` → `selection_score = -inf`
2. **A fixed multiplicative penalty**

       selection_score = ucb / (1 + (1 + cost/max_cost))

Measured against the seven arms' cold-start estimates, that penalty spans
`2.1296 .. 3.0000`, so the cheapest arm gets at most a **1.4087×** advantage
over the dearest. It is a bounded thumb on the scale, with no user control.

Latency is measured at [policy.py:159](policy.py#L159), emitted to the UI at
[policy.py:176](policy.py#L176), rendered at
[viz.html:975](static/viz.html#L975) — and then **discarded**. It is never
persisted (no column in `decisions`), never estimated, never used to select.
Verified: no latency field exists in `Algorithm/M91_AWS.json` or any
`Result/*.json` log either.

## 3. Decision: priority is NOT a context feature

The tempting move is to add priority as one-hot dims to `x`, letting the bandit
learn priority-conditional quality. **Rejected**, for two reasons.

**It has no signal to learn from.** `verify.check(task_type, text, expected,
schema)` does not take priority and never should — whether `\boxed{42}` is
correct is independent of whether the user was in a hurry. Reward is therefore
priority-independent *by construction*. Adding 7 arms × N dims of free
parameters for a function provably constant in that argument means spending
real pulls to discover it is constant.

**It destroys all learned state.** `CONTEXT_DIM` is baked into both persistence
layers. Verified:

    snapshot dim 49 != running 52                     ← bandit.restore raises
    operands could not be broadcast (52,52) (49,49)   ← stored vector into wide bandit

Every row in `decisions` holds a 49-d blob, so `pending_judge` would have to
discard all of them. Today that costs 10 pulls and 27 rows. At 10,000 pulls it
would not be affordable.

**Priority is a preference over the trade-off, not a feature of the world.** It
belongs in `select()`. `CONTEXT_DIM` stays 49; `bandit.npz` and every decision
row survive this change untouched.

(Features that genuinely belong in `x` are ones changing an answer's
*difficulty* — query length, detected language, whether the prompt contains
code. Out of scope here.)

## 4. Five objectives

| # | Objective | Source | Better when | Bound | Learned how |
|---|---|---|---|---|---|
| 1 | Quality | `bandit.score(x)` → `ucb` | high | UCB (already) | LinUCB `A`/`b` |
| 2 | Cost $ | `arms.token_cost` | low | LCB | `cost.py` (exists) |
| 3 | Latency ms | `policy.py:159` | low | LCB | **new** `latency.py` |
| 4 | Verbosity tokens | `usage.outputTokens` | low | LCB | **new**, same module |
| 5 | Format compliance | **new** `verify.format_ok` | high | **UCB** (Wilson) | **new**, same module |

### Direction of optimism

An unmeasured arm must look *attractive* so it gets tried at least once. Which
bound achieves that flips with the objective's direction:

- cost / latency / verbosity are **better low** → **LCB**, so an untried arm
  looks cheap / fast / terse and gets tried. This is exactly why `cost.py`
  already uses an LCB.
- format compliance is **better high** → **UCB**, so an untried arm looks
  compliant. Using an LCB here would make an unmeasured arm look
  non-compliant and it would *never* be tried — reintroducing the starvation
  bug `cost.py`'s LCB was written to avoid.

### Cost and verbosity are not the same axis

Worth stating because it looks like double-counting. Verified:

- Hold output tokens fixed at 500 for all seven arms: cost still spans
  `$0.00008400 .. $0.00064800`, a 7.7× spread. Same verbosity, different cost.
- A cheap rambler vs a dear terse arm: nova-micro at 2000 output tokens costs
  `$0.00029400`; llama3-3-70b at 150 costs `$0.00039600`. **Cost prefers the
  rambler, verbosity prefers the terse arm — they disagree.**

Verbosity is about the *answer* being tight; cost is about the *bill*. Related,
not collinear.

## 5. Format compliance without touching the verdict contract

`verify.check` returns `1` / `0` / `None`, and "`None` must never collapse into
`0`" is the most load-bearing invariant in the system. This design does not go
near it.

There are exactly **6** `None`-producing paths today (enumerated by running
every branch):

| task | situation | verdict | `format_ok` | at fault |
|---|---|---|---|---|
| math | `\boxed{42}`, expected 42 | `1` | `1` | — |
| math | `\boxed{41}`, expected 42 | `0` | `1` | arm: wrong maths, obeyed format |
| math | no `\boxed{}` at all | `None` | `0` | **arm: format miss** |
| math | `\boxed{@@@}` unparseable | `None` | `0` | **arm: format miss** |
| math | no `expected` supplied | `None` | `None` | verifier blind |
| code | fenced python | `1`/`0` | `1` | — |
| code | fenced rust | `None` | `0` | **arm: wrong language** |
| code | bare expression only | `None` | `0` | **arm: no fence** |
| structured | valid JSON | `1` | `1` | — |
| structured | invalid JSON | `0` | `0` | arm: both |
| prose | anything | `None` | `None` | verifier blind |

Today those collapse two very different things: *"the arm failed to format"*
(arm-attributable, worth learning) and *"the verifier cannot judge"* (nothing
about the arm at all).

**Add a separate function, leave `check` alone:**

    verify.format_ok(task_type, text, expected=None, schema=None) -> 1 | 0 | None

`format_ok` returns `None` **only** when the verifier is genuinely blind —
prose, or math with no `expected`. That recovers **4 of the 6** dead paths as
learnable signal.

Critically, `format_ok` feeds a *separate estimator*, never `A`/`b`. So:

- `verdict` still drives the cascade and the LinUCB update. Untouched.
- A math arm that never emits `\boxed{}` gets penalised **in selection**
  without the cascade ever treating its answer as wrong.

## 6. The maths: from division to scalarization

### Why the current rule cannot carry a second objective

**It does not extend.** Adding latency would mean
`ucb / (penalty_cost · penalty_latency)` or similar — relative weighting
becomes an emergent property of an arbitrary product, with no place to put a
user's weight that means anything.

**It is sign-unstable.** Verified:

    ucb=+1.0 → cheap +0.4696  vs expensive +0.3333  → cheap wins
    ucb= 0.0 → cheap  0.0000  vs expensive  0.0000  → tie
    ucb=-1.0 → cheap -0.4696  vs expensive -0.3333  → EXPENSIVE wins

Dividing a negative number by a *larger* penalty makes it *less* negative, so
the cost penalty inverts into a cost *bonus*.

In fairness, this is a **latent hazard, not an active bug**: sweeping for
negative `ucb` found `0 / 4000` probes after 600 mixed-reward pulls, because
`e` stays large enough to dominate. `p` alone does go negative (44.5% at 10
pulls, 0% by 50), but `ucb = p + e` did not. It is still one more reason not to
build on that shape.

### The replacement

Normalize each objective to `[0,1]` across the **eligible arms in that
attempt**, then take a weighted sum:

    score_a = w_q·q̃_a − w_c·c̃_a − w_ℓ·ℓ̃_a − w_v·ṽ_a + w_f·f̃_a

    with  Σw = 1,  w ≥ 0

    q̃_a = (ucb_a  − min ucb) / (max ucb − min ucb)      ↑ better
    c̃_a = (cost_a − min cost) / (max cost − min cost)    ↓ better
    ℓ̃_a = (lat_a  − min lat)  / (max lat  − min lat)     ↓ better
    ṽ_a = (out_a  − min out)  / (max out  − min out)     ↓ better
    f̃_a = (fmt_a  − min fmt)  / (max fmt  − min fmt)     ↑ better

Note the signs: four objectives are subtracted, format compliance is **added**,
matching its direction.

**Normalization is mandatory, not cosmetic.** Raw scales are incommensurable:

    quality ucb  ~ [0, 20]           (measured: e=19.8 cold, 0.67 warm)
    cost  $/call ~ [8.4e-5, 6.5e-4]
    latency ms   ~ [300, 3000]

A naive sum makes cost **1543×** too small to move the decision. Min-max also
keeps the weights behaving consistently whether the bandit is cold or warm.

**Subtraction is sign-stable** — a penalty stays a penalty at any `ucb` — and
extends to any number of objectives. `w` gains an interpretable meaning: *"I
will give up 30% of the quality range to save the full cost range."*

### Degeneracy guard — required, not defensive

Verified: min-max over a single value is a `ZeroDivisionError`, and that case
is **reachable on the last cascade attempt**, when only one arm remains
untried.

    if hi - lo < 1e-12:  normalized = 0.0 for every arm

All arms tie on that objective, so it drops out of the decision — which is
correct: an objective on which every candidate is identical carries no
information.

## 7. Presets

Named presets, validated exactly the way `task_type` already is
([app.py:62-67](app.py#L62-L67)).

    PRIORITIES = {            # (w_quality, w_cost, w_latency, w_verbosity, w_format)
        "quality":  (1.00, 0.00, 0.00, 0.00, 0.00),
        "balanced": (0.50, 0.20, 0.15, 0.05, 0.10),
        "cheap":    (0.35, 0.50, 0.00, 0.10, 0.05),
        "fast":     (0.35, 0.05, 0.50, 0.05, 0.05),
        "strict":   (0.40, 0.10, 0.05, 0.05, 0.40),
    }

Each tuple must sum to `1.0` — asserted at import, alongside
`assert_fallbacks_fit_budget`.

Against a plausible arm profile these select genuinely different arms:

| preset | winner |
|---|---|
| `quality` | llama3-3-70b |
| `balanced` | llama3-3-70b |
| `cheap` | nova-lite |
| `fast` | nova-lite |
| `strict` | llama3-3-70b |

**`balanced` is quality-leaning, not a midpoint — say so in the config
comment.** Sweeping `w_cost` shows the winner switches from llama3-3-70b to
nova-lite between `0.3` and `0.4`; `balanced` sits at `0.20`, below that
threshold. That is a defensible default, but it should be named honestly rather
than implied to be neutral. It also reflects a real dominance effect: when one
arm leads on *both* quality and format, every quality-leaning preset picks it.

## 8. Files

| File | Change |
|---|---|
| `router/latency.py` | **new** — `PerformanceEstimator`: latency LCB, verbosity LCB, format-compliance Wilson UCB. Per-`(arm, turn)` buckets mirroring `cost.py`. Snapshot `state/perf.npz`. |
| `router/verify.py` | **add** `format_ok`. `check` untouched. |
| `router/config.py` | `PRIORITIES`, weight-sum assertion, `TYPICAL_LATENCY_MS` / `TYPICAL_OUTPUT_TOKENS` fallbacks, Wilson `z`. |
| `router/bandit.py` | `select(x, candidates, costs, latencies, verbosities, formats, remaining_budget, weights)`. Replace the division with §6's scalarization. Keep budget pruning and `-inf`/`reason` exactly as-is — the UI depends on `reason`, and `json_safe` on the `-inf`. |
| `router/policy.py` | Thread `priority` through. Feed latency/verbosity/format samples **on the success path only**, mirroring the cost rule at [policy.py:182-189](policy.py#L182-L189): an errored call's 0 ms and 0 tokens are not samples. |
| `router/store.py` | ⚠️ **migration required** — see §9. Add `latency_ms`, `output_tokens`, `format_ok` columns. |
| `router/app.py` | `priority: str = "balanced"` on `CompleteRequest`, validated against `PRIORITIES`. Restore/snapshot the new estimator in `build_default_router` and `lifespan`. |
| `router/static/viz.html` | Priority dropdown; per-arm normalized terms and weighted contributions. |

## 9. The migration hazard

`store.py`'s `_SCHEMA` uses `CREATE TABLE IF NOT EXISTS`. Verified: re-running
an *edited* schema against an existing table **silently does nothing** —

    columns after re-running edited schema: ['a']    # 'b' never added

So simply adding columns to `_SCHEMA` will not touch the existing 27-row
`router.db`. It needs an explicit `PRAGMA table_info(decisions)` check plus
`ALTER TABLE decisions ADD COLUMN ...` for each missing column, run once at
`Store.__init__`. Without it the new columns exist for fresh databases only,
and the failure is invisible until an `INSERT` names a column that isn't there.

## 10. What is deliberately not changed

- `CONTEXT_DIM` stays **49**. No projection bump, no snapshot reset, no
  discarded rows.
- `verify.check`'s three-valued contract, byte for byte.
- The reward path: `verdict` drives cascade and LinUCB; `logged_reward = None
  if error else verdict`.
- Budget pruning stays a hard constraint ahead of scalarization. Priority
  changes *preference*, never affordability — `"quality"` must not authorise
  overspending.
- Single-worker `flock`, buffered responses, the `-inf`/`json_safe` contract.

## 11. Test obligations

- Each preset's weights sum to 1.0; unknown priority → 422.
- Scalarization: ordering is sign-stable at negative `ucb` (the case the
  division rule inverts).
- Degenerate normalization (one eligible arm; all-equal values) → no
  `ZeroDivisionError`, objective drops out.
- Optimism direction: an unmeasured arm is *tried* under each preset —
  specifically, an arm with no format samples is not starved.
- `format_ok` returns `None` for exactly prose and math-without-`expected`, and
  `0` for the four arm-attributable paths.
- Error path contributes no latency, verbosity, or format sample.
- Migration: an old 27-row DB gains the columns; a fresh DB is identical.
- `CONTEXT_DIM == 49` and the existing snapshot still restores.
- Default `priority="balanced"` reproduces sensible behaviour with an empty
  estimator (cold start).
