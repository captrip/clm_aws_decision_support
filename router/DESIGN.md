# Per-Turn LLM Router — Design

**Status:** approved design, not yet implemented
**Date:** 2026-07-31
**Scope:** local-only. This directory is excluded from git via `.git/info/exclude`.

A routing service that picks a Bedrock model per conversation turn, learns from observed
outcomes, and hides the choice behind one API. Derived from the LinUCB cascade in
[../Algorithm/Algo1_AWS.ipynb](../Algorithm/Algo1_AWS.ipynb) and
[../Algorithm/Algo2_AWS.ipynb](../Algorithm/Algo2_AWS.ipynb), adapted from
retry-within-a-question to turn-within-a-conversation.

---

## 1. Problem

Given a conversation and a new user turn, choose which of seven Bedrock models should
answer. Cheap models are fast and ~20× less expensive; expensive models succeed on harder
turns. The router should learn which turns need which arm, from live traffic, without
offline training or a simulator.

The decision must be fast enough to be invisible next to a multi-second model call:
target under 15ms.

### Why a contextual bandit

The alternative is a static rule ("code → big model, chitchat → small"). A bandit learns
the partition from outcomes instead, keeps adapting when arms or prices change, and — via
the context vector — can learn per-arm specialization rather than a single
difficulty axis.

### What differs from the notebooks

| | Notebooks | This router |
| :--- | :--- | :--- |
| Meaning of a "step" | Retry on the same question | A turn in a conversation |
| Reward source | Inline LLM grader vs ground truth | Verifier where possible, sampled judge otherwise |
| Context dim | 775 | 49 |
| Budget scope | Per question | Per conversation |
| Reward timing | Immediate | Immediate or delayed (async judge) |
| Runs | Notebook, batch | HTTP service, live |

---

## 2. Approach

Contextual bandit (LinUCB) over all seven arms, with Algo2's cost-penalized selection and
a hybrid cascade: turns with a programmatic verifier may silently escalate to a stronger
arm on failure; unverifiable turns commit their first answer and learn after the fact.

Two approaches were considered and rejected:

- **Difficulty predictor + static ladder** — one regression predicting whether the cheapest
  arm suffices, with a learned threshold. Converges much faster but collapses all arms onto
  one axis and cannot learn that one arm is better at code while another is better at
  extraction. Preferable only under very low traffic (< ~100 turns/day).
- **Two-level tier bandit** — a 2-arm small/large bandit, then cheapest-untried within the
  tier. Splits the difference but adds machinery and makes the tier boundary an arbitrary
  tunable.

**Known limitation, accepted:** with seven arms and a 49-d context, expect on the order of
a few thousand turns before routing clearly beats "always use a mid-tier arm." Cold-start
mitigation is deliberately out of scope for this iteration (§9).

---

## 3. Architecture

A single FastAPI process. One endpoint. Bandit matrices in memory, snapshotted to `.npz`;
decisions and rewards in SQLite.

### Single-worker invariant

**The service MUST run with exactly one worker process** (`uvicorn --workers 1`).

Each worker would hold its own copy of `A`/`b` and snapshot to the same file, so the last
writer silently discards the other's learning. The failure is invisible — the service keeps
serving, it just stops learning. `app.py` asserts single-worker at startup and refuses to
boot otherwise.

Outgrowing one worker means moving state to Redis, or keeping decisions on a local
read-only copy and applying reward updates through a single-consumer queue. Neither is in
this iteration.

### Modules

| Module | Responsibility | Depends on |
| :--- | :--- | :--- |
| `arms.py` | `MODELS_CONFIG` (7 arms + prices), `bedrock_converse()` with retry/backoff | boto3 |
| `context.py` | messages + turn state → 49-d vector | sentence-transformers |
| `bandit.py` | LinUCB `select()` / `update()` / snapshot / restore. Pure numpy | — |
| `cost.py` | per-(arm, turn) LCB cost estimate, budget pruning | — |
| `verify.py` | task type → verifier fn, or `None` if unverifiable | — |
| `policy.py` | the cascade: select → call → verify → escalate | all above |
| `store.py` | SQLite: decisions, pending rewards, snapshots | sqlite3 |
| `judge.py` | background sampled LLM judge → delayed reward | arms, store |
| `app.py` | HTTP, validation, response assembly | policy, store |

`arms.py` and the `bandit.py` core lift almost directly from `Algo1_AWS.ipynb`: same
Cholesky-with-`inv`-fallback score, same `A += xxᵀ` update, same `.npz` persistence, same
`bedrock_converse` throttle handling.

### Arms

From `Algo1_AWS.ipynb`, on-demand `us-east-1` prices per token:

| Arm | input | output |
| :--- | ---: | ---: |
| `amazon.nova-micro-v1:0` | 0.035/1e6 | 0.14/1e6 |
| `amazon.nova-lite-v1:0` | 0.06/1e6 | 0.24/1e6 |
| `mistral.mistral-7b-instruct-v0:2` | 0.15/1e6 | 0.20/1e6 |
| `us.meta.llama4-scout-17b-instruct-v1:0` | 0.17/1e6 | 0.66/1e6 |
| `us.meta.llama3-1-8b-instruct-v1:0` | 0.22/1e6 | 0.22/1e6 |
| `us.meta.llama4-maverick-17b-instruct-v1:0` | 0.24/1e6 | 0.97/1e6 |
| `us.meta.llama3-3-70b-instruct-v1:0` | 0.72/1e6 | 0.72/1e6 |

Judge: `us.anthropic.claude-sonnet-5` (cross-region inference profile).

All cost metrics derive from this table; it must be updated if arms or region change.

---

## 4. Context vector (49-d)

```
[0:32]   projected query embedding (MiniLM 384-d → 32-d)
[32]     turn index / MAX_TURNS
[33]     log10(prompt tokens) / 5
[34]     log10(conversation tokens so far) / 5
[35]     1.0 if a previous arm failed this turn, else 0.0
[36:43]  one-hot: last failed arm (7 arms)
[43:47]  one-hot: task type (code / structured / math / prose)
[47]     retry depth within this turn / MAX_CASCADE
[48]     bias, always 1.0
```

Features 35–42 carry over the notebooks' actual contribution
(`construct_feature_vector`): they let the router learn "Nova Micro already failed here,
skip Nova Lite entirely." Feature 48 gives each arm an unconditional baseline, so a good
arm can score above zero even when the projected embedding is near-orthogonal to
everything seen so far.

**Why 49 and not 775.** With 775 features per arm and realistically a few hundred
observations per arm, `θ` is dominated by the ridge regulariser and the confidence term —
the router explores near-randomly for a very long time. Cutting to 49 reduces
samples-to-converge by roughly an order of magnitude.

**Projection: random now, PCA later.** PCA needs traffic that does not exist yet, so start
with a fixed-seed random Gaussian projection (Johnson–Lindenstrauss preserves pairwise
distances adequately at 32 dims and needs zero data). Swap in a PCA fit on logged
embeddings once a few thousand turns have accumulated.

The swap invalidates all learned `A`/`b` — the feature space changes meaning. It is a
deliberate one-time reset, not something to iterate on. The projection matrix is versioned
in the snapshot; `bandit.py` refuses to restore a snapshot whose projection version does
not match the running one.

---

## 5. Decision path

```
request → context.build()           ~10ms   (embedding dominates)
        → cost.estimate(arm, turn)  ~0
        → bandit.select()           <1ms    (7 × 49×49 Cholesky)
        → arms.bedrock_converse()   seconds
        → verify.check()            0–50ms
        → if failed and retries left and budget allows: escalate, loop
        → store.log_decision()
        → response
```

~11ms of router overhead. Selection follows Algo2's `select_model_value_density`,
simplified:

1. Score every arm: `ucb = xᵀθ + α·√(xᵀA⁻¹x)`, with `α = 0.675`, `λ = 0.45` (notebook values).
2. Prune arms already tried this turn.
3. Prune arms whose estimated cost exceeds the conversation's remaining budget.
4. Among survivors, maximize `ucb / (1 + (1 + cost / max_eligible_cost))`.

If every arm is pruned, the turn fails with a budget-exhausted status rather than silently
calling something unaffordable.

Note the notebooks pass `REWARD_IMPORTANCE` / `COST_IMPORTANCE` into this function but
never use them (and `Algo2.ipynb` never defines them at all). They are dropped here; the
cost penalty is the normalization above.

### API contract

`POST /v1/complete` — opaque `model: "auto"`, no caller control over arm choice.

**All responses are buffered; there is no streaming.** Once tokens are streamed they cannot
be retracted, so streaming and silent escalation are incompatible. Buffering keeps one code
path and costs time-to-first-token, which is the accepted trade.

Response reports the arm that served, cascade depth, and cost, so callers can attribute
spend and debugging is possible. A `decision_id` is returned for later reward attribution.

---

## 6. Reward

Two sources, one update path.

### Verifier — immediate, dense

`verify.py` maps task type to a check returning `1`, `0`, or `None` ("cannot tell"):

| Task type | Check | Cost |
| :--- | :--- | :--- |
| `structured` | JSON parses and validates against the caller's schema | ~0 |
| `code` | extract block → `ast.parse` (Python) or `tsc --noEmit` | ~50ms |
| `math` | extract `\boxed{}` → sympy equivalence against expected, when provided | ~10ms |
| `prose` | none | — |

**Only turns with a non-`None` verdict cascade.** A `None` verdict commits the first answer
and waits on the judge.

### Sampled judge — delayed, sparse

`judge.py` pulls a configurable fraction (start: 10%) of unverified turns from a SQLite
queue and asks Claude Sonnet 5 whether the response adequately answered the request
(True/False). Runs as a background task, never in the request path.

Reuses the grader quirks already solved in the AWS notebooks: `thinking: disabled` (Sonnet 5
otherwise spends the whole token budget on `reasoningContent` and returns no text), no
`temperature` in `inferenceConfig`, and joining text across all content blocks rather than
assuming `content[0]`.

### The update, and its one subtlety

Per arm actually called: `A += xxᵀ`, `b += r·x`, with `r ∈ {0, 1}`.

**Each arm must be updated with the context vector it saw at selection time, not the final
one.** In a cascade, arm 1 is chosen at retry depth 0 (vector `x₀`); after it fails, arm 2
is chosen at depth 1 with a different vector `x₁` (failed-arm one-hot now set, depth
incremented). Updating arm 1 with `x₁` attributes to it a context that did not exist when
it was chosen, and corrupts learning in a way that is very hard to detect.

`store.py` therefore persists the exact vector alongside each decision, and `policy.py`
replays them on completion. This is a deliberate change from the notebooks, which reuse one
`x_i` per step — correct there only because exactly one arm runs per step.

Delayed judge rewards use the same mechanism: the vector is already in SQLite, so an update
arriving 30 seconds later is identical to an immediate one. Bandits are order-insensitive.

---

## 7. Cost and budget

`cost.py` is Algo2's `ModelCostEstimator`: running mean and variance of observed cost per
`(arm, turn_index)`, returning a lower confidence bound `mean − β·stderr` (β = 1.0), with a
fallback below two samples.

Keyed by turn index because prompt cost grows as the conversation accumulates — Algo3's
"positional awareness," which matters more for real conversations than it did for retries.

**Budget is per conversation, not per turn.** A running total decremented by actual spend,
so a cheap early exchange banks headroom for a hard question later.

### The fallback-cost trap

`Algo3.ipynb` is unusable cold: `DEFAULT_FALLBACK_COST = 0.01` against a test budget of
`0.00014217` — 70× too large, so nothing fits the knapsack and every test question exits at
zero steps. See [../README_AWS.md](../README_AWS.md) §"Notes on Algo3's test budget".

Here, each arm's fallback is derived from its own per-token price × a typical token count,
not one global constant. `app.py` asserts at startup that every arm's fallback estimate is
below the default conversation budget, so a misconfiguration fails loudly at boot instead of
silently routing to nothing.

---

## 8. Testing

The substantive logic is in pure functions, which is why the modules are split as they are.

- **`bandit.py`** — synthetic arms with known reward functions: assert regret shrinks over
  rounds; snapshot/restore round-trips exactly; Cholesky and `inv` paths agree; restore
  rejects a mismatched projection version.
- **`context.py`** — dimension is always 49; deterministic for identical input; failed-arm
  one-hot lands in the correct slot; task-type one-hot is mutually exclusive.
- **`cost.py`** — LCB below mean; converges to mean as N grows; fallback triggers below 2
  samples; boot assertion catches an over-large fallback.
- **`policy.py`** — Bedrock and verifier mocked: escalates on verifier failure; does not
  escalate on a `None` verdict; stops at max cascade depth; stops when budget is exhausted;
  **each arm is updated with its own context vector** (the §6 subtlety).
- **`store.py`** — decision round-trips including the vector; delayed reward attaches to the
  right decision.
- **Replay harness** — run [../Result/algo1res.json](../Result/algo1res.json) (500 questions
  with per-arm scores, chosen arm, cost, correctness) through the router offline to
  sanity-check accuracy and cost before any live traffic.

Bedrock is mocked everywhere except one opt-in smoke test making a real call.

---

## 9. Out of scope

- **Cold start.** Accepted as an open problem for this iteration. The warm-start option —
  replaying `Result/algo1res.json` to prime `A`/`b` — is available later but is off-policy
  and biased toward whatever the cascade happened to pick.
- **Prompt-cache affinity.** If a conversation's prefix is cached on the arm that served
  turn 1, routing turn 2 elsewhere discards that cache and pays full input price, so a
  nominally cheaper arm can cost more. The notebooks' `calculate_token_cost` models only
  input/output rates. A real gap; not addressed here.
- **Multi-worker state** (see §3 invariant).
- **Auth, rate limiting, multi-tenancy.**
- **Streaming** (see §5 — structurally incompatible with silent escalation).
