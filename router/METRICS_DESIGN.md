# Free metric selection + versioned basis expansion

Local prototype design. Not committed — `router/` is in `.git/info/exclude`.

Supersedes [PRIORITY_DESIGN.md](PRIORITY_DESIGN.md), which proposed five fixed
objectives and named presets. That design is subsumed: presets become defaults
over a *registry*, and the registry is open.

Extends [DESIGN.md](DESIGN.md). Read §4 (context vector), §6 (verify/judge),
§7 (cost) and [bandit.py](bandit.py) `select` first.

## 1. Goals

1. **Default metrics out of the box**, with users free to choose which metrics
   drive selection and how much each weighs.
2. **Learning adjusts accordingly** — but see §3, because there is exactly one
   safe reading of that.
3. **A migration path to non-linear models** that does not require a state
   reset when we take it.

## 2. Where user preference may and may not enter

The whole architecture turns on this one question.

### The rejected design: fold weights into the reward

Learning from `b += (w_q·r)·x` — i.e. letting the user's weights shape what the
bandit learns — was tested against a non-stationary traffic mix (quality-heavy
first half, cost-heavy second half). RMSE against a *specific* user's true
utility:

| test user | folded model | per-objective |
|---|---|---|
| quality (0.9/0.1) | 1.4902 | 0.000375 |
| balanced (0.5/0.5) | 0.5618 | 0.000321 |
| cheap (0.2/0.8) | 2.0771 | 0.000375 |

A folded model is only ever right for the *average historical user*, and cannot
be told which user it is serving. It is also non-stationary:

    ||θ_folded(first half) − θ_folded(second half)|| = 3.0575
    ||θ_quality(first half) − θ_quality(second half)|| = 0.0001

Because `b += r·x` accumulates forever, a month of cost-sensitive traffic
**permanently** biases the quality model. Old preferences never wash out.
**Rejected.**

Also rejected: *"only update the bandit when priority == quality"*. At 1.4
pulls/arm (§6) no data can be discarded.

### The design: learn every metric from every request

Preference enters **only at the argmax**, never in any estimator's update.
Per-objective estimation is essentially exact for *any* weight setting from the
same data — the 0.0003 column above.

**Consequence to accept honestly:** a metric is only learned for arms the
policy actually pulls. That is inherent to bandits — there are no
counterfactuals — and it is why every estimator must be fed on **every** pull,
and why §5 exists.

## 3. The metric registry

A metric is a **registered objective**, not user-supplied code:

    Metric = (name, extractor, direction, bound_type)

    extractor(arm, text, usage, latency_ms, verdict, task_type) -> float | None
    direction  : "higher_better" | "lower_better"
    bound_type : "lcb" | "ucb" | "wilson_ucb"

Users choose *which* registered metrics to weight and by how much. They do not
supply callables, for two reasons:

- An arbitrary callable would run inside `Router.complete` on Starlette's
  threadpool, where it could block, raise, or run unbounded — in the request
  path. `viz.py` already had to solve exactly this hazard (`emit` never blocks
  and never raises).
- A metric must be measured *consistently over history* to be estimable. One
  invented at request time has zero samples for every arm and contributes only
  noise.

### Built-in registry

| name | source | direction | bound | notes |
|---|---|---|---|---|
| `quality` | `bandit.score(x).ucb` | higher | UCB (LinUCB) | the only thing `A`/`b` learn |
| `cost` | `arms.token_cost` | lower | LCB | `cost.py` exists today |
| `latency` | `policy.py:159` | lower | LCB | **measured today and discarded** |
| `verbosity` | `usage.outputTokens` | lower | LCB | |
| `format` | `verify.format_ok` | higher | Wilson UCB | new; see §4 |

`latency_ms` is computed at [policy.py:159](policy.py#L159), emitted at
[policy.py:176](policy.py#L176), rendered at
[viz.html:975](static/viz.html#L975), then thrown away — no DB column, and
absent from `Algorithm/M91_AWS.json` and every `Result/*.json`.

### Direction of optimism is per metric, not global

An unmeasured arm must look *attractive* so it gets tried. Which bound achieves
that flips with direction: `lower_better` → **LCB**; `higher_better` → **UCB**.
Using an LCB on `format` would make an unmeasured arm look non-compliant and it
would never be tried — the starvation bug `cost.py`'s LCB was written to avoid.

`format` is a Bernoulli rate, so **Wilson**, not a naive mean: `0/1` gives mean
`0.0`, condemning an arm on one observation, where Wilson upper gives `0.5`.

### Weight semantics

- **Renormalize, don't reject.** `{quality: 2, cost: 1}` → `{0.667, 0.333}`.
  Weights are *ratios*, which is what "cost twice as important" means, and it
  keeps scores in a bounded comparable range across different metric sets.
- **Unspecified metrics get weight 0, not their default.** Otherwise "I only
  care about these two" is silently not honoured. Defaults apply only when *no*
  weights are supplied at all.
- Metrics are composable: each is normalized independently, so adding a third
  metric does not perturb the other two's numbers.

### API

    POST /v1/complete
    { ..., "priority": "cheap" }                       # named default
    { ..., "weights": {"quality": 2, "latency": 1} }   # free choice, renormalized

`weights` wins when both are present. Unknown metric name → 422.

## 4. Format compliance without touching the verdict contract

`verify.check` returns `1`/`0`/`None`, and "`None` must never collapse into
`0`" is the most load-bearing invariant here. This goes nowhere near it.

Running every branch found exactly **6** `None`-producing paths, collapsing two
different things: *the arm failed to format* (arm-attributable, learnable) and
*the verifier cannot judge* (nothing about the arm).

Add a **separate** function; leave `check` byte-identical:

    verify.format_ok(task_type, text, expected=None, schema=None) -> 1 | 0 | None

| task | situation | verdict | `format_ok` |
|---|---|---|---|
| math | `\boxed{41}`, expected 42 | `0` | `1` (wrong maths, obeyed format) |
| math | no `\boxed{}` | `None` | **`0`** |
| math | `\boxed{@@@}` | `None` | **`0`** |
| math | no `expected` | `None` | `None` (blind) |
| code | fenced rust | `None` | **`0`** |
| code | bare expression | `None` | **`0`** |
| prose | anything | `None` | `None` (blind) |

`format_ok` is `None` only when the verifier is genuinely blind, recovering
**4 of 6** dead paths. It feeds its own estimator, never `A`/`b`: a math arm
that never emits `\boxed{}` is penalised *in selection* while the cascade still
never treats its answer as wrong.

## 5. Scalarization, and the starvation hazard it creates

### Scoring

Normalize each metric to `[0,1]` across the **eligible arms in that attempt**,
then take the weighted sum, adding `higher_better` terms and subtracting
`lower_better` ones:

    score_a = Σ_m  ±w_m · m̃_a          Σw = 1, w ≥ 0

Normalization is **mandatory**: raw scales are incommensurable (`ucb ~ [0,20]`
vs `cost ~ [8.4e-5, 6.5e-4]`), and a naive sum makes cost **1543×** too small
to move the decision.

**Degeneracy guard, required not defensive:** min-max over a single value is a
`ZeroDivisionError`, and that case is **reachable on the last cascade attempt**
when one arm remains untried. `if hi − lo < 1e-12: normalized = 0.0` for all —
a metric on which every candidate is identical carries no information.

### Evidence shrinkage — min-max alone is not enough

Plain min-max has a defect that only shows up once weights are user-controlled:
**it always spans exactly 1.0, whether the spread is real or noise.** Two arms
whose `ucb` differ by `0.0001` normalize to `0.0` and `1.0` — identical to two
arms differing by `2.0`. Measured, with a 7.7× cost gap in play:

| quality gap | `balanced` picks |
|---|---|
| 0.0001 (pure noise) | the **dearer** arm |
| 0.30 (real) | the dearer arm |

A meaningless quality edge overrides a 7.7× price difference, and it does not
self-correct: the raw spread grows with pulls (0.24 cold → 0.68 hot) but
normalized always spans 1.0.

So scale each metric's contribution by whether its spread is distinguishable
from that metric's own noise:

    m̃_a = minmax(m)_a · k        k = spread / (spread + noise_m)

For `quality`, `noise` is the natural one the algorithm already computes — the
median `e` across eligible arms. For the sample-mean metrics it is the pooled
standard error. `k → 0` when the spread is noise, `k → 1` when it dwarfs it.
Verified (arm0 dear-but-better, arm1 cheap-but-worse, `balanced`):

| quality gap | median `e` | k | pick |
|---|---|---|---|
| 0.0001 | 0.67 | 0.0001 | cheap arm |
| 0.05 | 0.67 | 0.069 | cheap arm |
| 0.30 | 0.67 | 0.309 | cheap arm |
| 0.30 | **0.20** | 0.600 | **dear-but-better arm** |
| 1.00 | 0.67 | 0.599 | dear-but-better arm |

Quality overrides cost only once its gap is large *relative to the confidence
width* — and a tighter `e` flips the decision sooner, which is exactly the
right coupling. On the live cold state (`k = 0.137` at 1.4 pulls/arm) quality is
almost weightless and `balanced` picks `nova-micro`, matching today's
production pick; with plain min-max it picked `llama3-1-8b` off a 0.12 `ucb`
spread between arms with **zero pulls between them**.

`k` is per-metric and multiplies only that metric's term, so the weights keep
their meaning; it must be reported on the `scored` event or a term looking
"too small" in the UI is unexplainable.

This replaces `ucb / (1 + (1 + cost/max_cost))`, which cannot carry a second
objective (relative weighting becomes an emergent property of an arbitrary
product) and is sign-unstable: at `ucb = −1.0`, cheap scores `−0.4696` vs
expensive `−0.3333`, inverting the cost penalty into a cost *bonus*. In
fairness that is **latent, not active** — `0/4000` probes reached negative
`ucb` after 600 mixed-reward pulls, because `e` dominates. Still not a shape to
build on.

### The hazard: permanent starvation

With `w_cost = 0.65` vs `w_quality = 0.35`, the maximum quality advantage
(0.35) is *less than* the cost disadvantage (0.65), so the best arm can **never
win**, however good it becomes. Measured over 3000 turns:

| ε | pulls | unmeasured arms | max estimate error |
|---|---|---|---|
| 0.00 | `[2998, 1, 1, 0, 0]` | 2 | 0.700 |
| 0.02 | `[2876, 71, 23, 17, 13]` | 0 | 0.178 |
| 0.05 | `[2659, 224, 59, 31, 27]` | 0 | 0.063 |

Executing the user's preference is correct; freezing the arm's estimates
forever is not — a price cut or model upgrade could never be noticed.

**LinUCB's `e` term cannot fix this.** It explores in *quality* space and
cannot overcome a hard weight deficit in the *scalarized* score. So:

    EXPLORE_EPSILON = 0.02   # ~2% of turns pick uniformly among ELIGIBLE arms

Four integration requirements, each verified:

1. **Order is fixed:** prune (candidate? affordable?) → `return None` if nothing
   eligible → *then* ε-floor or argmax, both over the **same** eligible list.
   The floor must never reach a pruned arm: that would overspend the budget and
   break "priority changes preference, never affordability".
2. **Injected rng.** `LinUCB.__init__` currently takes no rng, and `select` has
   no randomness, so tests are exact. Add `rng=None`; tests pass a seeded rng
   or `eps=0.0`. **At `eps=0.0` the branch is unreachable and every existing
   selection test passes unchanged** — that is the regression gate.
3. **New `reason` value.** `viz.html` renders `info["reason"]`. A forced pick
   must be labelled `"forced exploration"` with a `forced` boolean on the
   `scored` event, or the UI shows an arm winning with a visibly non-maximal
   `selection_score` and looks broken.
4. Budget pruning and the `-inf`/`json_safe` contract stay exactly as-is.

## 6. Non-linear bandits: right instinct, premature by two orders of magnitude

### The reward genuinely is non-linear

The interaction the notebooks already rely on — *"arm A failed on this query,
so its near-twin will too"* — is `x[35]` (failure flag) × `x[0:32]`
(embedding), i.e. degree-2. A linear model must score these additively and is
helpless:

    LINEAR fit to a pure interaction (XOR):  RMSE 0.4992, corr 0.0557
                                             (predicting the mean gives 0.5000)
    WITH the product term added:              RMSE 0.0018, corr 1.0000

And it pays. Cumulative regret with a true degree-2 interaction:

| T | linear | +1 product term | ratio |
|---|---|---|---|
| 100 | 42.1 | 44.0 | 0.96× |
| 250 | 93.6 | 74.3 | 1.26× |
| 500 | 159.4 | 109.2 | 1.46× |
| 4000 | 645.3 | 252.0 | **2.56×** |

**Note the crossover: expansion LOSES until T≈250 and only pays past ~500.**

### Why not now

Parameters per arm:

| model | params/arm |
|---|---|
| LinUCB today | **49** |
| + full quadratic | 1,274 |
| neural, 1×64 hidden | 3,264 |

The live snapshot holds **10 total pulls across all 7 arms — 1.4 per arm,
against 49 parameters already.** LinUCB regret is `O(d√T)`, scaling with the
parameter count.

Worse, below ~50 pulls/arm the confidence width dominates entirely:

| pulls | `e` |
|---|---|
| 10 | 20.90 |
| 50 | 1.03 |
| 500 | ~0.20 |

With `e ≈ 20.9` against `p ≈ 1`, **the bandit is not yet using its quality
model at all**, so a richer model class changes nothing today.

Expansion is also not free: wider `φ` means wider intervals (at n=50, `e` =
2.02 at d=49 vs 7.37 at d=117). It trades bias for variance and is paid for in
extra pulls.

### The middle path: targeted basis expansion, shipped OFF

LinUCB is linear in the **features**, not the raw input. Define

    φ(x) = [ x , x[35]·x[0:32] , x[43:47]·x[35] , x[47]·x[0:32] ]

and run the *identical* algorithm on `φ(x)`: `A += φφᵀ`, `b += r·φ`,
`p = φ·θ`, `e = α√(φᵀA⁻¹φ)`. Cholesky scoring, the ridge closed form,
order-insensitive updates, delayed rewards and snapshots all survive untouched.

| expansion | +dims | total d | vs base |
|---|---|---|---|
| `any_failed × embedding` | 32 | 81 | 1.65× |
| `task × any_failed` | 4 | 53 | 1.08× |
| `retry_depth × embedding` | 32 | 81 | 1.65× |
| **all three** | 68 | **117** | **2.4×** |
| full quadratic | 1225 | 1274 | 26× |
| neural 1×64 | — | — | 66× |

**Ship with `φ = identity`.** Verified byte-identical to today: `A`/`b`
identical across all 7 arms and identical scores. Flip terms on via config once
past ~500 pulls/arm.

    PHI_VERSION = "identity-v1"
    PHI_TERMS   = []            # off today

## 7. Replay: the migration mechanism

**Correction to an earlier claim of my own.** I first concluded that widening
`φ` preserves learned state losslessly by embedding
`A' = [[A, 0], [0, λI]]`. That was a tautology — it only showed "embedding
equals assuming the new dims were zero." For *product* features that is plainly
false:

    ||A_embed − A_true|| = 153.7691
    ||b_embed − b_true|| =   4.7406

The real mechanism is **replay**. `store.py` stores **raw `x`** plus reward, so
`φ` can be recomputed for every rewarded row and `A`/`b` rebuilt exactly.
Requirement: `φ` must be a pure function of stored raw `x`, and both
`policy.complete` and `record_reward` must apply it.

### Two versions, composing asymmetrically

| bump | stored rows | recovery |
|---|---|---|
| `PROJECTION_VERSION` (meaning of raw `x`) | invalid | full reset, as today |
| `PHI_VERSION` (meaning of `φ(x)`) | **still valid** | **replay rebuilds `A`/`b` exactly** |

That asymmetry is the whole reason to add `φ` versioning now: it converts future
model-class changes from resets into replays. The snapshot must check both.

### Replay is also crash recovery — an unplanned finding

Measured against the live state:

    rewarded rows in DB : 18
    pulls in snapshot   : 10

The snapshot only persists on graceful shutdown
([app.py:141-148](app.py#L141-L148)), so any turn after the last clean exit is
in the DB but not the `.npz`. File mtimes confirm the DB is newer
(`2026-08-06 02:50` vs `2026-08-03 18:09`), and **7 rewarded rows postdate the
snapshot** — real learning silently lost.

(Caveat on that split: SQLite `CURRENT_TIMESTAMP` is UTC while mtime is local,
so the per-side counts are approximate. The robust claim needs no timezone
reasoning: the DB holds strictly more learnable observations than the snapshot
has pulls, so replay is a **superset** rebuild and should be treated as
authoritative, regenerating the snapshot from it.)

So expose it: `POST /admin/rebuild` — replay all rewarded rows at the current
`PROJECTION_VERSION`, rebuild `A`/`b`, write a fresh snapshot. Useful for a
`φ` bump, after a crash, and as a consistency check.

## 8. Files

| File | Change |
|---|---|
| `router/metrics.py` | **new** — registry, `Metric`, `MetricEstimator` (LCB / UCB / Wilson per-`(arm, turn)` buckets mirroring `cost.py`), weight parsing + renormalization. Snapshot `state/metrics.npz`. |
| `router/features.py` | **new** — `phi(x)`, `PHI_TERMS`, `phi_dim()`. Pure function of raw `x`. |
| `router/verify.py` | **add** `format_ok`. `check` untouched. |
| `router/config.py` | `METRIC_DEFAULTS` (the old presets), `EXPLORE_EPSILON = 0.02`, `PHI_VERSION`, `PHI_TERMS`, Wilson `z`, weight-sum assertions at import. |
| `router/bandit.py` | `select(..., metric_values, weights, rng)`: scalarization + ε-floor + `"forced exploration"` reason. Apply `phi` in `score`/`update`. Snapshot checks `PHI_VERSION`. `rng=None` in `__init__`. |
| `router/policy.py` | Thread `weights` through. Feed every metric estimator on the success path only, mirroring the cost rule at [policy.py:182-189](policy.py#L182-L189) — an errored call's 0 ms / 0 tokens are not samples. |
| `router/store.py` | ⚠️ **migration** — `_SCHEMA` uses `CREATE TABLE IF NOT EXISTS`; verified that re-running an edited schema **silently adds nothing**. Needs `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` for `latency_ms`, `output_tokens`, `format_ok`. |
| `router/app.py` | `priority` / `weights` on `CompleteRequest`; `POST /admin/rebuild`; restore/snapshot the metric estimator. |
| `router/static/viz.html` | Metric weight controls; per-arm normalized terms and weighted contributions; render `"forced exploration"`. |

## 9. Invariants that do not change

- `CONTEXT_DIM` stays **49**; `PROJECTION_VERSION` is not bumped; no decision
  row is discarded.
- `verify.check`'s three-valued contract, byte for byte.
- `verdict` drives cascade and LinUCB; `logged_reward = None if error else
  verdict`.
- Budget pruning is a hard constraint ahead of scalarization **and** ahead of
  the ε-floor. Preference never authorises overspending.
- Single-worker `flock`; buffered responses; `-inf` → `json_safe` → `null`.
- `φ = identity` and `eps = 0.0` reproduce today's behaviour exactly.

## 10. Test obligations

- **Regression gate:** with `PHI_TERMS = []` and `eps = 0.0`, every existing
  bandit/policy test passes unchanged.
- `φ = identity` gives byte-identical `A`/`b` and scores (verified).
- Scalarization ordering is sign-stable at negative `ucb`.
- Degenerate normalization (one eligible arm; all-equal values) → no
  `ZeroDivisionError`; the metric drops out.
- ε-floor never selects a budget-pruned or already-tried arm.
- ε-floor is deterministic under a seeded rng; `"forced exploration"` reaches
  the `scored` event.
- Optimism direction per metric: an arm with no `format` samples is not starved.
- `format_ok` returns `None` for exactly prose and math-without-`expected`; `0`
  for the four arm-attributable paths.
- Error path contributes no latency / verbosity / format sample.
- Weights renormalize; unspecified metric ⇒ weight 0; unknown name ⇒ 422.
- Replay: rebuilding from the DB reproduces `A`/`b` exactly for a given `φ`;
  a `PHI_VERSION` mismatch is refused by `restore`.
- Migration: an existing 27-row DB gains the columns; a fresh DB is identical.
