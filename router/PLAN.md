# Per-Turn LLM Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI service that picks one of seven Bedrock models per conversation turn using a LinUCB contextual bandit, silently escalates on verifier failure, and learns from verifier and sampled-judge rewards.

**Architecture:** Single-worker FastAPI process. Bandit matrices (`A`, `b` per arm) live in memory and snapshot to `.npz`; decisions, rewards, and conversation spend live in SQLite. One buffered endpoint, `POST /v1/complete`. Pure-numpy bandit, pure-function context builder and cost estimator, so the substantive logic is testable without network access.

**Tech Stack:** Python 3.13, FastAPI, uvicorn, numpy, boto3 (Bedrock `converse`), sentence-transformers (all-MiniLM-L6-v2), sympy, pytest, httpx.

**Spec:** [DESIGN.md](DESIGN.md). Section references below (§3, §6…) point there.

## Global Constraints

- **Nothing is committed.** `router/` is excluded via `.git/info/exclude`. Every task ends with a verification step, not a `git commit`. Do not `git add` anything under `router/`, and do not amend the exclude file.
- **All files live under `router/`.** Do not create or modify files elsewhere in the repo.
- **Interpreter:** `/opt/anaconda3/bin/python3.13`. The system `python3` is 3.9.6 and has none of the dependencies. Create the venv with `/opt/anaconda3/bin/python3.13 -m venv router/.venv`.
- **Context dimension is exactly 49.** Layout fixed by §4: `[0:32]` projected embedding, `[32]` turn index, `[33]` prompt tokens, `[34]` conversation tokens, `[35]` failure flag, `[36:43]` last-failed-arm one-hot (7), `[43:47]` task-type one-hot (4), `[47]` retry depth, `[48]` bias.
- **Seven arms**, exact Bedrock IDs and on-demand `us-east-1` prices per §3. Do not substitute models.
- **LinUCB constants:** `ALPHA = 0.675`, `LAMBDA_REG = 0.45`, `BETA_COST = 1.0` (from the notebooks).
- **Single worker.** The service must refuse to boot as a second process (§3 invariant).
- **Each arm is updated with the context vector it saw at selection time** (§6). This is the one correctness rule that fails silently if broken.
- **No streaming.** All responses buffered (§5).
- **AWS profile** `258574424891`, region `us-east-1`, matching `Algorithm/Algo1_AWS.ipynb`.
- **Bedrock is mocked in all tests** except the single opt-in smoke test in Task 11.

---

## File Structure

| File | Responsibility |
| :--- | :--- |
| `router/config.py` | All constants: arms, prices, dims, hyperparameters, budgets, fallback costs |
| `router/bandit.py` | `LinUCB`: `score`, `select`, `update`, `snapshot`, `restore`. Pure numpy |
| `router/context.py` | `ContextBuilder.build()` → 49-d vector |
| `router/cost.py` | `CostEstimator`: per-(arm, turn) LCB cost |
| `router/verify.py` | `check()` → `1` / `0` / `None` |
| `router/store.py` | `Store`: SQLite decisions, rewards, conversation spend |
| `router/arms.py` | `bedrock_converse()`, `token_cost()` |
| `router/policy.py` | `Router.complete()` — the cascade |
| `router/judge.py` | `run_judge_batch()` — delayed reward |
| `router/app.py` | FastAPI app, single-worker lock, boot assertions |
| `router/replay.py` | Offline replay harness over `Result/algo1res.json` |
| `router/tests/test_*.py` | One test module per source module |

Task order follows the dependency graph: config → bandit / context / cost / verify / store (independent) → arms → policy → app → judge → replay.

---

### Task 1: Project scaffolding and config

**Files:**
- Create: `router/requirements.txt`, `router/config.py`, `router/__init__.py`, `router/tests/__init__.py`
- Test: `router/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ARMS: dict[str, dict[str, float]]`, `ARM_NAMES: list[str]`, `N_ARMS: int`, `CONTEXT_DIM: int`, `EMBED_DIM`, `PROJ_DIM`, `ALPHA`, `LAMBDA_REG`, `BETA_COST`, `MAX_TURNS`, `MAX_CASCADE`, `TASK_TYPES: list[str]`, `DEFAULT_CONVERSATION_BUDGET: float`, `TYPICAL_PROMPT_TOKENS`, `TYPICAL_OUTPUT_TOKENS`, `PROJECTION_VERSION: str`, `JUDGE_MODEL`, `JUDGE_SAMPLE_RATE`, `AWS_PROFILE`, `AWS_REGION`, `fallback_cost(arm: str) -> float`, `assert_fallbacks_fit_budget() -> None`.

- [ ] **Step 1: Create the venv and install dependencies**

```bash
/opt/anaconda3/bin/python3.13 -m venv router/.venv
router/.venv/bin/pip install --quiet fastapi uvicorn numpy boto3 sentence-transformers sympy pytest httpx
router/.venv/bin/python -c "import fastapi, numpy, boto3, sentence_transformers, sympy, pytest, httpx; print('deps ok')"
```

Expected: `deps ok`. All subsequent commands use `router/.venv/bin/python` and `router/.venv/bin/pytest`.

- [ ] **Step 2: Write `router/requirements.txt`**

```
fastapi
uvicorn
numpy
boto3
sentence-transformers
sympy
pytest
httpx
```

- [ ] **Step 3: Create empty package markers**

```bash
touch router/__init__.py router/tests/__init__.py
```

- [ ] **Step 4: Write the failing test**

`router/tests/test_config.py`:

```python
import pytest

from router import config


def test_seven_arms_in_price_order():
    assert len(config.ARM_NAMES) == 7
    assert config.N_ARMS == 7
    assert config.ARM_NAMES[0] == "amazon.nova-micro-v1:0"
    assert config.ARM_NAMES[-1] == "us.meta.llama3-3-70b-instruct-v1:0"
    # Cheapest-first by input price, as in Algo1_AWS.ipynb
    inputs = [config.ARMS[a]["input_cost"] for a in config.ARM_NAMES]
    assert inputs == sorted(inputs)


def test_every_arm_has_both_prices():
    for arm in config.ARM_NAMES:
        assert config.ARMS[arm]["input_cost"] > 0
        assert config.ARMS[arm]["output_cost"] > 0


def test_context_dim_matches_layout():
    # 32 embedding + 1 turn + 1 prompt tok + 1 conv tok + 1 fail flag
    # + 7 last-failed-arm + 4 task type + 1 retry depth + 1 bias
    assert config.PROJ_DIM == 32
    assert config.CONTEXT_DIM == 32 + 1 + 1 + 1 + 1 + 7 + 4 + 1 + 1
    assert config.CONTEXT_DIM == 49


def test_task_types_are_the_four_from_the_spec():
    assert config.TASK_TYPES == ["code", "structured", "math", "prose"]


def test_notebook_hyperparameters():
    assert config.ALPHA == 0.675
    assert config.LAMBDA_REG == 0.45
    assert config.BETA_COST == 1.0


def test_fallback_cost_is_per_arm_and_price_ordered():
    cheap = config.fallback_cost("amazon.nova-micro-v1:0")
    dear = config.fallback_cost("us.meta.llama3-3-70b-instruct-v1:0")
    assert 0 < cheap < dear


def test_fallback_costs_fit_the_default_budget():
    # The Algo3 trap: a global fallback 70x the budget means nothing ever fits
    # and every turn exits at zero steps. Must fail loudly instead.
    config.assert_fallbacks_fit_budget()


def test_assert_fallbacks_raises_when_budget_too_small(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_CONVERSATION_BUDGET", 1e-12)
    with pytest.raises(ValueError, match="fallback cost"):
        config.assert_fallbacks_fit_budget()
```

- [ ] **Step 5: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.config'`

- [ ] **Step 6: Write `router/config.py`**

```python
"""Constants for the per-turn router. See DESIGN.md sections 3, 4, 7."""

# --- Arms: Bedrock on-demand us-east-1 prices per token, cheapest input first.
# Copied from Algorithm/Algo1_AWS.ipynb. Update if arms or region change --
# every cost metric in the service derives from this table.
ARMS = {
    "amazon.nova-micro-v1:0": {"input_cost": 0.035 / 1e6, "output_cost": 0.14 / 1e6},
    "amazon.nova-lite-v1:0": {"input_cost": 0.06 / 1e6, "output_cost": 0.24 / 1e6},
    "mistral.mistral-7b-instruct-v0:2": {"input_cost": 0.15 / 1e6, "output_cost": 0.20 / 1e6},
    "us.meta.llama4-scout-17b-instruct-v1:0": {"input_cost": 0.17 / 1e6, "output_cost": 0.66 / 1e6},
    "us.meta.llama3-1-8b-instruct-v1:0": {"input_cost": 0.22 / 1e6, "output_cost": 0.22 / 1e6},
    "us.meta.llama4-maverick-17b-instruct-v1:0": {"input_cost": 0.24 / 1e6, "output_cost": 0.97 / 1e6},
    "us.meta.llama3-3-70b-instruct-v1:0": {"input_cost": 0.72 / 1e6, "output_cost": 0.72 / 1e6},
}
ARM_NAMES = list(ARMS.keys())
N_ARMS = len(ARM_NAMES)

# --- AWS access
AWS_PROFILE = "258574424891"
AWS_REGION = "us-east-1"

# Cross-region inference profile ID. Sonnet 5 has no plain on-demand ID.
JUDGE_MODEL = "us.anthropic.claude-sonnet-5"
JUDGE_SAMPLE_RATE = 0.10

# --- Context vector (DESIGN.md section 4)
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
PROJ_DIM = 32
TASK_TYPES = ["code", "structured", "math", "prose"]
CONTEXT_DIM = PROJ_DIM + 1 + 1 + 1 + 1 + N_ARMS + len(TASK_TYPES) + 1 + 1  # == 49

# Bumping this invalidates every learned A/b: the feature space changes meaning.
# bandit.restore() refuses a snapshot whose version differs. Changing from the
# seeded random projection to a PCA fit is a deliberate one-time reset.
PROJECTION_VERSION = "random-jl-v1"
PROJECTION_SEED = 20260731

# --- LinUCB (notebook values)
ALPHA = 0.675
LAMBDA_REG = 0.45

# --- Cost estimation (DESIGN.md section 7)
BETA_COST = 1.0
EPSILON_COST = 1e-10
MIN_SAMPLES_FOR_LCB = 2

# --- Cascade and budget
MAX_TURNS = 20
MAX_CASCADE = 3
DEFAULT_CONVERSATION_BUDGET = 0.02

# Token counts used to derive each arm's cold-start cost estimate.
TYPICAL_PROMPT_TOKENS = 400
TYPICAL_OUTPUT_TOKENS = 500

# --- Paths
SNAPSHOT_PATH = "router/state/bandit.npz"
COST_STATE_PATH = "router/state/cost.npz"
DB_PATH = "router/state/router.db"
LOCK_PATH = "router/state/router.lock"


def fallback_cost(arm: str) -> float:
    """Cold-start cost estimate for an arm with too few samples.

    Derived from the arm's own prices rather than one global constant. A single
    global fallback is what makes Algo3 unusable cold: its 0.01 against a
    0.00014217 budget means nothing ever fits and every turn exits at 0 steps.
    """
    cfg = ARMS[arm]
    return (
        TYPICAL_PROMPT_TOKENS * cfg["input_cost"]
        + TYPICAL_OUTPUT_TOKENS * cfg["output_cost"]
    )


def assert_fallbacks_fit_budget() -> None:
    """Fail loudly at boot if any arm's cold-start estimate exceeds the budget.

    Otherwise every arm is pruned before selection and the service silently
    routes to nothing.
    """
    for arm in ARM_NAMES:
        est = fallback_cost(arm)
        if est > DEFAULT_CONVERSATION_BUDGET:
            raise ValueError(
                f"{arm}: fallback cost ${est:.8f} exceeds default conversation "
                f"budget ${DEFAULT_CONVERSATION_BUDGET:.8f}. Every arm would be "
                f"pruned. Raise DEFAULT_CONVERSATION_BUDGET or lower "
                f"TYPICAL_*_TOKENS."
            )
```

- [ ] **Step 7: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_config.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 8: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output at all (the directory is excluded).

---

### Task 2: LinUCB bandit

**Files:**
- Create: `router/bandit.py`
- Test: `router/tests/test_bandit.py`

**Interfaces:**
- Consumes: `config.ARM_NAMES`, `config.CONTEXT_DIM`, `config.ALPHA`, `config.LAMBDA_REG`, `config.PROJECTION_VERSION`.
- Produces:
  - `class LinUCB(arm_names: list[str], dim: int, alpha: float, lambda_reg: float, projection_version: str)`
  - `.score(x: np.ndarray) -> dict[str, dict[str, float]]` — per arm `{"p": float, "e": float, "ucb": float}`
  - `.select(x, candidates: list[str], costs: dict[str, float], remaining_budget: float) -> tuple[str | None, dict]` — returns chosen arm (or `None` if all pruned) and the full score dict with `eligible` / `selection_score` annotations
  - `.update(arm: str, x: np.ndarray, reward: float) -> None`
  - `.snapshot(path: str) -> None`, `.restore(path: str) -> bool`

- [ ] **Step 1: Write the failing test**

`router/tests/test_bandit.py`:

```python
import numpy as np
import pytest

from router import config
from router.bandit import LinUCB


def make_bandit(dim=4, arms=("a", "b", "c")):
    return LinUCB(
        arm_names=list(arms),
        dim=dim,
        alpha=config.ALPHA,
        lambda_reg=config.LAMBDA_REG,
        projection_version="test-v1",
    )


def test_cold_start_scores_are_equal_across_arms():
    bandit = make_bandit()
    x = np.ones(4)
    scores = bandit.score(x)
    ucbs = [s["ucb"] for s in scores.values()]
    assert len(set(round(u, 12) for u in ucbs)) == 1
    # No reward seen yet, so the mean term is zero and score is pure exploration
    assert all(s["p"] == 0.0 for s in scores.values())
    assert all(s["e"] > 0 for s in scores.values())


def test_update_raises_the_rewarded_arm_above_the_others():
    bandit = make_bandit()
    x = np.array([1.0, 0.0, 0.0, 1.0])
    for _ in range(10):
        bandit.update("a", x, 1.0)
        bandit.update("b", x, 0.0)
    scores = bandit.score(x)
    assert scores["a"]["p"] > scores["b"]["p"]
    assert scores["a"]["ucb"] > scores["b"]["ucb"]


def test_confidence_term_shrinks_as_an_arm_is_pulled():
    bandit = make_bandit()
    x = np.array([1.0, 1.0, 0.0, 0.0])
    before = bandit.score(x)["a"]["e"]
    for _ in range(20):
        bandit.update("a", x, 1.0)
    after = bandit.score(x)["a"]["e"]
    assert after < before


def test_regret_shrinks_on_a_synthetic_problem():
    """Arm 'good' pays 1 when feature 0 is on; 'bad' never pays."""
    bandit = make_bandit(dim=3, arms=("good", "bad"))
    rng = np.random.default_rng(0)
    first_half_optimal = 0
    second_half_optimal = 0
    for step in range(400):
        x = np.array([1.0, rng.normal(), 1.0])
        chosen, _ = bandit.select(x, ["good", "bad"], {"good": 0.0, "bad": 0.0}, 1.0)
        reward = 1.0 if chosen == "good" else 0.0
        bandit.update(chosen, x, reward)
        if chosen == "good":
            if step < 200:
                first_half_optimal += 1
            else:
                second_half_optimal += 1
    assert second_half_optimal > first_half_optimal
    assert second_half_optimal > 150  # mostly exploiting by the second half


def test_select_prunes_arms_over_budget():
    bandit = make_bandit()
    x = np.ones(4)
    costs = {"a": 10.0, "b": 0.1, "c": 20.0}
    chosen, scores = bandit.select(x, ["a", "b", "c"], costs, remaining_budget=1.0)
    assert chosen == "b"
    assert scores["a"]["eligible"] is False
    assert scores["b"]["eligible"] is True


def test_select_returns_none_when_everything_is_pruned():
    bandit = make_bandit()
    x = np.ones(4)
    costs = {"a": 10.0, "b": 10.0, "c": 10.0}
    chosen, _ = bandit.select(x, ["a", "b", "c"], costs, remaining_budget=0.001)
    assert chosen is None


def test_select_respects_the_candidate_list():
    bandit = make_bandit()
    x = np.ones(4)
    costs = {"a": 0.1, "b": 0.1, "c": 0.1}
    chosen, _ = bandit.select(x, ["c"], costs, remaining_budget=1.0)
    assert chosen == "c"


def test_cost_penalty_prefers_the_cheaper_of_two_equal_arms():
    bandit = make_bandit()
    x = np.ones(4)
    # Identical UCB (cold start), so the cost penalty decides.
    chosen, _ = bandit.select(
        x, ["a", "b"], {"a": 1.0, "b": 0.01}, remaining_budget=10.0
    )
    assert chosen == "b"


def test_snapshot_restore_round_trips(tmp_path):
    bandit = make_bandit()
    x = np.array([0.5, 1.0, 0.0, 2.0])
    for _ in range(5):
        bandit.update("a", x, 1.0)
    path = str(tmp_path / "state.npz")
    bandit.snapshot(path)

    fresh = make_bandit()
    assert fresh.restore(path) is True
    np.testing.assert_allclose(fresh.models["a"]["A"], bandit.models["a"]["A"])
    np.testing.assert_allclose(fresh.models["a"]["b"], bandit.models["a"]["b"])
    assert fresh.score(x)["a"]["ucb"] == pytest.approx(bandit.score(x)["a"]["ucb"])


def test_restore_rejects_a_mismatched_projection_version(tmp_path):
    bandit = make_bandit()
    path = str(tmp_path / "state.npz")
    bandit.snapshot(path)

    other = LinUCB(
        arm_names=["a", "b", "c"],
        dim=4,
        alpha=config.ALPHA,
        lambda_reg=config.LAMBDA_REG,
        projection_version="different-v2",
    )
    with pytest.raises(ValueError, match="projection version"):
        other.restore(path)


def test_restore_returns_false_when_no_snapshot_exists(tmp_path):
    bandit = make_bandit()
    assert bandit.restore(str(tmp_path / "missing.npz")) is False


def test_singular_matrix_falls_back_without_raising():
    bandit = make_bandit()
    # Force a non-positive-definite A so Cholesky fails and inv is used.
    bandit.models["a"]["A"] = np.zeros((4, 4))
    scores = bandit.score(np.ones(4))
    assert "ucb" in scores["a"]
    assert np.isfinite(scores["a"]["ucb"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_bandit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.bandit'`

- [ ] **Step 3: Write `router/bandit.py`**

```python
"""LinUCB over model arms. Pure numpy, no I/O beyond snapshot files.

Lifted from LinUCBModel in Algorithm/Algo1_AWS.ipynb: same Cholesky-with-inv
fallback scoring, same A += xx^T / b += r*x update, same .npz persistence.
Selection adds Algo2's cost penalty and budget pruning.
"""

import os

import numpy as np

from router import config


class LinUCB:
    def __init__(
        self,
        arm_names,
        dim=config.CONTEXT_DIM,
        alpha=config.ALPHA,
        lambda_reg=config.LAMBDA_REG,
        projection_version=config.PROJECTION_VERSION,
    ):
        self.arm_names = list(arm_names)
        self.dim = dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.projection_version = projection_version
        self.models = {
            arm: {
                "A": np.identity(dim) * lambda_reg,
                "b": np.zeros(dim),
                "pulls": 0,
            }
            for arm in self.arm_names
        }

    # --- scoring

    def score(self, x):
        """Per-arm mean, confidence width, and their sum."""
        x = np.asarray(x, dtype=float)
        scores = {}
        for arm in self.arm_names:
            model = self.models[arm]
            try:
                chol = np.linalg.cholesky(model["A"])
                theta = np.linalg.solve(model["A"], model["b"])
                z = np.linalg.solve(chol, x)
                width = self.alpha * float(np.sqrt(np.sum(z ** 2)))
                mean = float(x.dot(theta))
            except np.linalg.LinAlgError:
                try:
                    a_inv = np.linalg.inv(model["A"])
                    theta = a_inv.dot(model["b"])
                    width = self.alpha * float(np.sqrt(x.dot(a_inv).dot(x)))
                    mean = float(x.dot(theta))
                except np.linalg.LinAlgError:
                    mean, width = 0.0, 0.0
            scores[arm] = {"p": mean, "e": width, "ucb": mean + width}
        return scores

    def select(self, x, candidates, costs, remaining_budget):
        """Pick an arm: prune by budget, then maximise cost-penalised UCB.

        Follows Algo2's select_model_value_density, minus the REWARD_IMPORTANCE
        and COST_IMPORTANCE parameters that notebook accepts but never uses.
        Returns (arm_or_None, annotated_scores).
        """
        scores = self.score(x)
        candidate_set = set(candidates)

        eligible = []
        for arm, info in scores.items():
            info["cost_estimate"] = float(costs.get(arm, 0.0))
            if arm not in candidate_set:
                info["eligible"] = False
                info["reason"] = "not a candidate"
                info["selection_score"] = float("-inf")
            elif info["cost_estimate"] > remaining_budget:
                info["eligible"] = False
                info["reason"] = "over budget"
                info["selection_score"] = float("-inf")
            else:
                info["eligible"] = True
                info["reason"] = "eligible"
                eligible.append(arm)

        if not eligible:
            return None, scores

        max_cost = max(scores[a]["cost_estimate"] for a in eligible)
        max_cost = max(max_cost, config.EPSILON_COST)

        best_arm, best_score = None, float("-inf")
        for arm in eligible:
            info = scores[arm]
            ratio = (info["cost_estimate"] + config.EPSILON_COST) / (
                max_cost + config.EPSILON_COST
            )
            penalty = 1.0 + (1.0 + ratio)
            info["selection_score"] = info["ucb"] / penalty
            if info["selection_score"] > best_score:
                best_score = info["selection_score"]
                best_arm = arm
        return best_arm, scores

    # --- learning

    def update(self, arm, x, reward):
        x = np.asarray(x, dtype=float)
        model = self.models[arm]
        model["A"] += np.outer(x, x)
        model["b"] += x * float(reward)
        model["pulls"] += 1

    # --- persistence

    def snapshot(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "__projection_version__": np.array(self.projection_version),
            "__dim__": np.array(self.dim),
        }
        for arm, model in self.models.items():
            payload[f"A::{arm}"] = model["A"]
            payload[f"b::{arm}"] = model["b"]
            payload[f"pulls::{arm}"] = np.array(model["pulls"])
        np.savez_compressed(path, **payload)

    def restore(self, path):
        """Load a snapshot. False if absent; raises on a version mismatch."""
        if not os.path.exists(path):
            return False
        loaded = np.load(path, allow_pickle=False)

        saved_version = str(loaded["__projection_version__"])
        if saved_version != self.projection_version:
            raise ValueError(
                f"snapshot projection version {saved_version!r} != running "
                f"{self.projection_version!r}. The feature space changed meaning, "
                f"so learned A/b are invalid. Delete {path} to start fresh."
            )
        saved_dim = int(loaded["__dim__"])
        if saved_dim != self.dim:
            raise ValueError(f"snapshot dim {saved_dim} != running {self.dim}")

        for arm in self.arm_names:
            if f"A::{arm}" in loaded and f"b::{arm}" in loaded:
                self.models[arm]["A"] = loaded[f"A::{arm}"]
                self.models[arm]["b"] = loaded[f"b::{arm}"]
                if f"pulls::{arm}" in loaded:
                    self.models[arm]["pulls"] = int(loaded[f"pulls::{arm}"])
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_bandit.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 3: Context builder

**Files:**
- Create: `router/context.py`
- Test: `router/tests/test_context.py`

**Interfaces:**
- Consumes: `config.EMBED_MODEL`, `config.EMBED_DIM`, `config.PROJ_DIM`, `config.CONTEXT_DIM`, `config.ARM_NAMES`, `config.TASK_TYPES`, `config.MAX_TURNS`, `config.MAX_CASCADE`, `config.PROJECTION_SEED`.
- Produces:
  - `class ContextBuilder(embedder=None, seed=config.PROJECTION_SEED)` — `embedder` may be any object with `.encode(str) -> np.ndarray`; when `None` the real SentenceTransformer is loaded lazily on first `build()`.
  - `.build(query: str, turn_index: int, prompt_tokens: int, conversation_tokens: int, task_type: str, failed_arms: list[str], retry_depth: int) -> np.ndarray` (shape `(49,)`)
  - `.projection: np.ndarray` (shape `(384, 32)`)

Note the test injects a fake embedder so no model download is needed. Task 11's smoke test exercises the real one.

- [ ] **Step 1: Write the failing test**

`router/tests/test_context.py`:

```python
import numpy as np
import pytest

from router import config
from router.context import ContextBuilder


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer."""

    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return rng.normal(size=config.EMBED_DIM)


def make_builder():
    return ContextBuilder(embedder=FakeEmbedder())


def build_default(builder, **overrides):
    kwargs = dict(
        query="what is 2+2",
        turn_index=0,
        prompt_tokens=10,
        conversation_tokens=10,
        task_type="prose",
        failed_arms=[],
        retry_depth=0,
    )
    kwargs.update(overrides)
    return builder.build(**kwargs)


def test_vector_is_always_49_dimensional():
    builder = make_builder()
    x = build_default(builder)
    assert x.shape == (config.CONTEXT_DIM,)
    assert x.shape == (49,)


def test_identical_input_gives_identical_vector():
    builder = make_builder()
    a = build_default(builder)
    b = build_default(builder)
    np.testing.assert_allclose(a, b)


def test_projection_shape_and_determinism_across_instances():
    one = ContextBuilder(embedder=FakeEmbedder())
    two = ContextBuilder(embedder=FakeEmbedder())
    assert one.projection.shape == (config.EMBED_DIM, config.PROJ_DIM)
    np.testing.assert_allclose(one.projection, two.projection)


def test_bias_feature_is_always_one():
    builder = make_builder()
    x = build_default(builder)
    assert x[48] == 1.0


def test_turn_index_is_normalised_and_clipped():
    builder = make_builder()
    assert build_default(builder, turn_index=0)[32] == 0.0
    mid = build_default(builder, turn_index=config.MAX_TURNS // 2)[32]
    assert 0.4 < mid < 0.6
    # Past MAX_TURNS the feature saturates rather than exceeding 1
    assert build_default(builder, turn_index=config.MAX_TURNS * 5)[32] == 1.0


def test_failure_flag_is_off_with_no_failures_and_on_with_one():
    builder = make_builder()
    assert build_default(builder, failed_arms=[])[35] == 0.0
    x = build_default(builder, failed_arms=[config.ARM_NAMES[0]])
    assert x[35] == 1.0


def test_last_failed_arm_one_hot_lands_in_the_right_slot():
    builder = make_builder()
    for idx, arm in enumerate(config.ARM_NAMES):
        x = build_default(builder, failed_arms=[arm])
        onehot = x[36:43]
        assert onehot.sum() == 1.0
        assert onehot[idx] == 1.0


def test_only_the_most_recent_failure_is_encoded():
    builder = make_builder()
    first, second = config.ARM_NAMES[0], config.ARM_NAMES[3]
    x = build_default(builder, failed_arms=[first, second])
    onehot = x[36:43]
    assert onehot.sum() == 1.0
    assert onehot[3] == 1.0


def test_task_type_one_hot_is_mutually_exclusive():
    builder = make_builder()
    for idx, task in enumerate(config.TASK_TYPES):
        x = build_default(builder, task_type=task)
        onehot = x[43:47]
        assert onehot.sum() == 1.0
        assert onehot[idx] == 1.0


def test_unknown_task_type_raises():
    builder = make_builder()
    with pytest.raises(ValueError, match="task_type"):
        build_default(builder, task_type="poetry")


def test_unknown_failed_arm_raises():
    builder = make_builder()
    with pytest.raises(ValueError, match="arm"):
        build_default(builder, failed_arms=["gpt-4"])


def test_retry_depth_is_normalised():
    builder = make_builder()
    assert build_default(builder, retry_depth=0)[47] == 0.0
    deep = build_default(builder, retry_depth=config.MAX_CASCADE)[47]
    assert deep == 1.0


def test_token_features_grow_with_token_counts():
    builder = make_builder()
    small = build_default(builder, prompt_tokens=10, conversation_tokens=10)
    large = build_default(builder, prompt_tokens=5000, conversation_tokens=50000)
    assert large[33] > small[33]
    assert large[34] > small[34]


def test_zero_tokens_does_not_produce_nan_or_inf():
    builder = make_builder()
    x = build_default(builder, prompt_tokens=0, conversation_tokens=0)
    assert np.all(np.isfinite(x))


def test_embedder_is_only_called_once_per_build():
    fake = FakeEmbedder()
    builder = ContextBuilder(embedder=fake)
    build_default(builder)
    assert fake.calls == 1


def test_all_features_are_finite_for_a_realistic_call():
    builder = make_builder()
    x = build_default(
        builder,
        query="write a function that reverses a linked list",
        turn_index=3,
        prompt_tokens=850,
        conversation_tokens=4200,
        task_type="code",
        failed_arms=[config.ARM_NAMES[1]],
        retry_depth=1,
    )
    assert np.all(np.isfinite(x))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.context'`

- [ ] **Step 3: Write `router/context.py`**

```python
"""Builds the 49-d context vector. Layout is fixed by DESIGN.md section 4.

Carries over construct_feature_vector from Algorithm/Algo1_AWS.ipynb -- notably
the last-failed-arm one-hot, which is what lets the bandit learn "the cheap arm
already failed here, skip its neighbour" -- at 49 dims instead of 775.
"""

import math

import numpy as np

from router import config

_ARM_INDEX = {arm: i for i, arm in enumerate(config.ARM_NAMES)}
_TASK_INDEX = {task: i for i, task in enumerate(config.TASK_TYPES)}


def _build_projection(seed):
    """Seeded Johnson-Lindenstrauss projection, 384 -> 32.

    Random rather than PCA because PCA needs traffic that does not exist yet.
    Swapping in a PCA fit means bumping config.PROJECTION_VERSION, which
    invalidates all learned A/b -- a deliberate one-time reset.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(config.PROJ_DIM),
        size=(config.EMBED_DIM, config.PROJ_DIM),
    )


class ContextBuilder:
    def __init__(self, embedder=None, seed=config.PROJECTION_SEED):
        self._embedder = embedder
        self.projection = _build_projection(seed)

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._embedder

    def build(
        self,
        query,
        turn_index,
        prompt_tokens,
        conversation_tokens,
        task_type,
        failed_arms,
        retry_depth,
    ):
        if task_type not in _TASK_INDEX:
            raise ValueError(
                f"unknown task_type {task_type!r}; expected one of {config.TASK_TYPES}"
            )
        failed_arms = list(failed_arms or [])
        for arm in failed_arms:
            if arm not in _ARM_INDEX:
                raise ValueError(f"unknown arm {arm!r} in failed_arms")

        x = np.zeros(config.CONTEXT_DIM, dtype=float)

        embedding = np.asarray(self.embedder.encode(query), dtype=float).reshape(-1)
        x[0 : config.PROJ_DIM] = embedding @ self.projection

        x[32] = min(turn_index / config.MAX_TURNS, 1.0)
        x[33] = math.log10(1.0 + max(prompt_tokens, 0)) / 5.0
        x[34] = math.log10(1.0 + max(conversation_tokens, 0)) / 5.0

        if failed_arms:
            x[35] = 1.0
            x[36 + _ARM_INDEX[failed_arms[-1]]] = 1.0

        x[43 + _TASK_INDEX[task_type]] = 1.0
        x[47] = min(retry_depth / config.MAX_CASCADE, 1.0)
        x[48] = 1.0
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_context.py -v`
Expected: PASS, 16 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 4: Cost estimator

**Files:**
- Create: `router/cost.py`
- Test: `router/tests/test_cost.py`

**Interfaces:**
- Consumes: `config.ARM_NAMES`, `config.MAX_TURNS`, `config.BETA_COST`, `config.MIN_SAMPLES_FOR_LCB`, `config.EPSILON_COST`, `config.fallback_cost`.
- Produces:
  - `class CostEstimator(arm_names=config.ARM_NAMES)`
  - `.update(arm: str, turn_index: int, actual_cost: float) -> None`
  - `.estimate(arm: str, turn_index: int) -> float` — LCB, or the arm's fallback below `MIN_SAMPLES_FOR_LCB`
  - `.estimate_all(turn_index: int) -> dict[str, float]`
  - `.snapshot(path: str) -> None`, `.restore(path: str) -> bool`

- [ ] **Step 1: Write the failing test**

`router/tests/test_cost.py`:

```python
import numpy as np
import pytest

from router import config
from router.cost import CostEstimator

ARM = config.ARM_NAMES[0]
DEAR_ARM = config.ARM_NAMES[-1]


def test_cold_estimate_is_the_per_arm_fallback():
    est = CostEstimator()
    assert est.estimate(ARM, 0) == pytest.approx(config.fallback_cost(ARM))
    assert est.estimate(DEAR_ARM, 0) == pytest.approx(config.fallback_cost(DEAR_ARM))
    # Per-arm, not one global constant -- this is the Algo3 trap
    assert est.estimate(ARM, 0) < est.estimate(DEAR_ARM, 0)


def test_one_sample_still_uses_the_fallback():
    est = CostEstimator()
    est.update(ARM, 0, 0.5)
    assert est.estimate(ARM, 0) == pytest.approx(config.fallback_cost(ARM))


def test_two_samples_switch_to_the_lcb():
    est = CostEstimator()
    est.update(ARM, 0, 0.010)
    est.update(ARM, 0, 0.020)
    assert est.estimate(ARM, 0) != pytest.approx(config.fallback_cost(ARM))


def test_lcb_is_at_or_below_the_mean():
    est = CostEstimator()
    for cost in [0.01, 0.02, 0.03, 0.04]:
        est.update(ARM, 0, cost)
    assert est.estimate(ARM, 0) <= 0.025


def test_lcb_converges_toward_the_mean_as_samples_grow():
    est_few = CostEstimator()
    est_many = CostEstimator()
    rng = np.random.default_rng(7)
    for _ in range(4):
        est_few.update(ARM, 0, float(rng.normal(0.02, 0.005)))
    rng = np.random.default_rng(7)
    for _ in range(400):
        est_many.update(ARM, 0, float(rng.normal(0.02, 0.005)))
    gap_few = abs(0.02 - est_few.estimate(ARM, 0))
    gap_many = abs(0.02 - est_many.estimate(ARM, 0))
    assert gap_many < gap_few


def test_estimate_is_always_positive():
    est = CostEstimator()
    for cost in [0.0, 0.0, 0.0]:
        est.update(ARM, 0, cost)
    assert est.estimate(ARM, 0) > 0


def test_statistics_are_kept_per_turn_index():
    """Prompt cost grows with conversation length, so turn 0 and turn 5 differ."""
    est = CostEstimator()
    for _ in range(5):
        est.update(ARM, 0, 0.001)
        est.update(ARM, 5, 0.010)
    assert est.estimate(ARM, 0) < est.estimate(ARM, 5)


def test_unseen_turn_falls_back_to_overall_statistics():
    est = CostEstimator()
    for _ in range(6):
        est.update(ARM, 0, 0.004)
    # Turn 9 has no samples of its own but the arm does overall.
    assert est.estimate(ARM, 9) == pytest.approx(0.004, rel=0.2)


def test_estimate_all_covers_every_arm():
    est = CostEstimator()
    everything = est.estimate_all(0)
    assert set(everything) == set(config.ARM_NAMES)
    assert all(v > 0 for v in everything.values())


def test_unknown_arm_raises():
    est = CostEstimator()
    with pytest.raises(KeyError):
        est.update("nope", 0, 0.1)


def test_snapshot_restore_round_trips(tmp_path):
    est = CostEstimator()
    for cost in [0.01, 0.02, 0.03]:
        est.update(ARM, 2, cost)
    path = str(tmp_path / "cost.npz")
    est.snapshot(path)

    fresh = CostEstimator()
    assert fresh.restore(path) is True
    assert fresh.estimate(ARM, 2) == pytest.approx(est.estimate(ARM, 2))


def test_restore_returns_false_when_absent(tmp_path):
    est = CostEstimator()
    assert est.restore(str(tmp_path / "missing.npz")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_cost.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.cost'`

- [ ] **Step 3: Write `router/cost.py`**

```python
"""Per-(arm, turn) cost estimation via a lower confidence bound.

This is Algo2's ModelCostEstimator. Keyed by turn index because prompt cost
grows as a conversation accumulates -- Algo3's "positional awareness", which
matters more across real turns than it did across retries.

Uses an LCB rather than the mean so an arm whose cost is still uncertain looks
cheap and gets tried, instead of being pruned before it has any samples.
"""

import json
import math
import os

import numpy as np

from router import config


def _empty():
    return {"count": 0, "total": 0.0, "sum_sq": 0.0}


class CostEstimator:
    def __init__(self, arm_names=None):
        self.arm_names = list(arm_names or config.ARM_NAMES)
        self.stats = {
            arm: {"overall": _empty(), "by_turn": {}} for arm in self.arm_names
        }

    def update(self, arm, turn_index, actual_cost):
        if arm not in self.stats:
            raise KeyError(f"unknown arm {arm!r}")
        cost = float(actual_cost)
        key = str(int(turn_index))
        per_turn = self.stats[arm]["by_turn"].setdefault(key, _empty())
        for bucket in (per_turn, self.stats[arm]["overall"]):
            bucket["count"] += 1
            bucket["total"] += cost
            bucket["sum_sq"] += cost * cost

    def _lcb(self, bucket):
        if bucket["count"] < config.MIN_SAMPLES_FOR_LCB:
            return None
        n = bucket["count"]
        mean = bucket["total"] / n
        variance = max(0.0, (bucket["sum_sq"] / n) - mean * mean)
        std_error = math.sqrt(variance) / math.sqrt(n)
        return max(config.EPSILON_COST, mean - config.BETA_COST * std_error)

    def estimate(self, arm, turn_index):
        if arm not in self.stats:
            raise KeyError(f"unknown arm {arm!r}")
        key = str(int(turn_index))
        per_turn = self.stats[arm]["by_turn"].get(key)
        if per_turn is not None:
            value = self._lcb(per_turn)
            if value is not None:
                return value
        value = self._lcb(self.stats[arm]["overall"])
        if value is not None:
            return value
        return config.fallback_cost(arm)

    def estimate_all(self, turn_index):
        return {arm: self.estimate(arm, turn_index) for arm in self.arm_names}

    def snapshot(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, stats=np.array(json.dumps(self.stats)))

    def restore(self, path):
        if not os.path.exists(path):
            return False
        loaded = np.load(path, allow_pickle=False)
        saved = json.loads(str(loaded["stats"]))
        for arm in self.arm_names:
            if arm in saved:
                self.stats[arm] = saved[arm]
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_cost.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 5: Verifiers

**Files:**
- Create: `router/verify.py`
- Test: `router/tests/test_verify.py`

**Interfaces:**
- Consumes: `config.TASK_TYPES`.
- Produces:
  - `check(task_type: str, response_text: str, expected: str | None = None, schema: dict | None = None) -> int | None` — `1` pass, `0` fail, `None` cannot tell (which means no cascade)
  - `extract_code_block(text: str) -> tuple[str | None, str | None]` → `(language, code)`
  - `extract_boxed(text: str) -> str | None`

- [ ] **Step 1: Write the failing test**

`router/tests/test_verify.py`:

```python
from router import verify


# --- prose: never verifiable, so it must never trigger a cascade

def test_prose_always_returns_none():
    assert verify.check("prose", "Paris is the capital of France.") is None
    assert verify.check("prose", "") is None


# --- structured

def test_valid_json_matching_schema_passes():
    schema = {"type": "object", "required": ["name", "age"]}
    assert verify.check("structured", '{"name": "Ada", "age": 36}', schema=schema) == 1


def test_json_in_a_fenced_block_is_still_found():
    schema = {"type": "object", "required": ["name"]}
    text = 'Here you go:\n```json\n{"name": "Ada"}\n```\n'
    assert verify.check("structured", text, schema=schema) == 1


def test_malformed_json_fails():
    schema = {"type": "object", "required": ["name"]}
    assert verify.check("structured", '{"name": ', schema=schema) == 0


def test_json_missing_a_required_key_fails():
    schema = {"type": "object", "required": ["name", "age"]}
    assert verify.check("structured", '{"name": "Ada"}', schema=schema) == 0


def test_structured_without_a_schema_only_checks_parseability():
    assert verify.check("structured", '{"anything": 1}') == 1
    assert verify.check("structured", "not json at all") == 0


def test_structured_accepts_a_top_level_array():
    assert verify.check("structured", "[1, 2, 3]") == 1


# --- code

def test_syntactically_valid_python_passes():
    text = "```python\ndef add(a, b):\n    return a + b\n```"
    assert verify.check("code", text) == 1


def test_syntactically_invalid_python_fails():
    text = "```python\ndef add(a, b)\n    return a + b\n```"
    assert verify.check("code", text) == 0


def test_python_without_a_fence_is_still_checked():
    assert verify.check("code", "x = 1\ny = x + 2\n") == 1


def test_code_with_no_extractable_block_returns_none():
    assert verify.check("code", "I would use a hash map for this.") is None


def test_unsupported_language_returns_none():
    text = "```rust\nfn main() { println!(\"hi\"); }\n```"
    assert verify.check("code", text) is None


def test_extract_code_block_reports_the_language():
    lang, code = verify.extract_code_block("```python\nx = 1\n```")
    assert lang == "python"
    assert code.strip() == "x = 1"


def test_extract_code_block_returns_none_none_when_absent():
    assert verify.extract_code_block("just prose") == (None, None)


# --- math

def test_matching_boxed_answer_passes():
    assert verify.check("math", r"So the answer is \boxed{42}.", expected="42") == 1


def test_wrong_boxed_answer_fails():
    assert verify.check("math", r"The answer is \boxed{41}.", expected="42") == 0


def test_algebraically_equivalent_forms_pass():
    assert verify.check("math", r"\boxed{1/2}", expected="0.5") == 1
    assert verify.check("math", r"\boxed{2/4}", expected="1/2") == 1


def test_missing_boxed_answer_returns_none():
    assert verify.check("math", "The answer is forty two.", expected="42") is None


def test_math_without_an_expected_answer_returns_none():
    assert verify.check("math", r"\boxed{42}") is None


def test_unparseable_expression_returns_none_not_zero():
    """Cannot tell is different from wrong: it must not trigger a cascade."""
    assert verify.check("math", r"\boxed{\text{see above}}", expected="42") is None


def test_extract_boxed_takes_the_last_box():
    assert verify.extract_boxed(r"first \boxed{1} then \boxed{2}") == "2"


def test_extract_boxed_handles_nested_braces():
    assert verify.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


# --- unknown task type

def test_unknown_task_type_returns_none():
    assert verify.check("poetry", "roses are red") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_verify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.verify'`

- [ ] **Step 3: Write `router/verify.py`**

```python
"""Programmatic verifiers. DESIGN.md section 6.

Three return values, and the distinction matters:
  1    -> verified good
  0    -> verified bad, escalate to a stronger arm
  None -> cannot tell, so commit this answer and wait for the sampled judge

Returning 0 where None is meant would cascade on unverifiable output and burn
money for nothing.
"""

import ast
import json
import re

_FENCE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)
_SUPPORTED_CODE_LANGS = {"python", "py", None}


def extract_code_block(text):
    """Return (language, code) from the first fenced block, or (None, None)."""
    if not text:
        return None, None
    match = _FENCE.search(text)
    if match:
        lang = (match.group(1) or "").lower() or None
        return lang, match.group(2)
    return None, None


def extract_boxed(text):
    r"""Return the contents of the last \boxed{...}, brace-balanced."""
    if not text:
        return None
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip()


def _extract_json(text):
    """Parse JSON from raw text or a fenced block. Returns (ok, value)."""
    if not text:
        return False, None
    candidates = []
    _, fenced = extract_code_block(text)
    if fenced:
        candidates.append(fenced)
    candidates.append(text)
    for candidate in candidates:
        stripped = candidate.strip()
        try:
            return True, json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
    return False, None


def _check_structured(text, schema):
    ok, value = _extract_json(text)
    if not ok:
        return 0
    if not schema:
        return 1
    if schema.get("type") == "object":
        if not isinstance(value, dict):
            return 0
        for key in schema.get("required", []):
            if key not in value:
                return 0
    elif schema.get("type") == "array" and not isinstance(value, list):
        return 0
    return 1


def _check_code(text):
    lang, code = extract_code_block(text)
    if code is None:
        # No fence. Treat the whole response as Python only if it parses as
        # more than a bare expression, otherwise we cannot tell.
        try:
            tree = ast.parse(text or "")
        except SyntaxError:
            return None
        if not tree.body or all(
            isinstance(node, ast.Expr) for node in tree.body
        ):
            return None
        return 1
    if lang not in _SUPPORTED_CODE_LANGS:
        return None
    try:
        ast.parse(code)
        return 1
    except SyntaxError:
        return 0


def _check_math(text, expected):
    if expected is None:
        return None
    answer = extract_boxed(text)
    if answer is None:
        return None
    if answer.strip() == str(expected).strip():
        return 1
    import sympy

    try:
        got = sympy.sympify(answer.replace("\\", ""))
        want = sympy.sympify(str(expected).replace("\\", ""))
        return 1 if sympy.simplify(got - want) == 0 else 0
    except Exception:
        # Unparseable: cannot tell, not wrong.
        return None


def check(task_type, response_text, expected=None, schema=None):
    if task_type == "structured":
        return _check_structured(response_text, schema)
    if task_type == "code":
        return _check_code(response_text)
    if task_type == "math":
        return _check_math(response_text, expected)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_verify.py -v`
Expected: PASS, 23 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 6: SQLite store

**Files:**
- Create: `router/store.py`
- Test: `router/tests/test_store.py`

**Interfaces:**
- Consumes: `config.CONTEXT_DIM`.
- Produces:
  - `class Store(db_path: str)` — creates tables on construction
  - `.log_decision(conversation_id: str, turn_index: int, arm: str, context_vector: np.ndarray, retry_depth: int, task_type: str, cost: float, reward: int | None, needs_judge: bool, query: str, response_text: str) -> int` (decision id)
  - `.get_decision(decision_id: int) -> dict | None` — `context_vector` comes back as `np.ndarray`
  - `.attach_reward(decision_id: int, reward: int) -> None`
  - `.pending_judge(limit: int) -> list[dict]`
  - `.get_spent(conversation_id: str) -> float`
  - `.add_spend(conversation_id: str, cost: float) -> None`
  - `.close() -> None`

Storing the vector per decision is what makes §6 enforceable: an update arriving later still uses the vector its arm saw.

- [ ] **Step 1: Write the failing test**

`router/tests/test_store.py`:

```python
import numpy as np
import pytest

from router import config
from router.store import Store

ARM = config.ARM_NAMES[0]


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    yield s
    s.close()


def make_vector(fill=0.5):
    return np.full(config.CONTEXT_DIM, fill)


def log_one(store, **overrides):
    kwargs = dict(
        conversation_id="conv-1",
        turn_index=0,
        arm=ARM,
        context_vector=make_vector(),
        retry_depth=0,
        task_type="prose",
        cost=0.001,
        reward=None,
        needs_judge=True,
        query="hello",
        response_text="hi there",
    )
    kwargs.update(overrides)
    return store.log_decision(**kwargs)


def test_log_decision_returns_increasing_ids(store):
    first = log_one(store)
    second = log_one(store)
    assert isinstance(first, int)
    assert second > first


def test_decision_round_trips_including_the_vector(store):
    vector = np.linspace(0, 1, config.CONTEXT_DIM)
    decision_id = log_one(store, context_vector=vector, cost=0.0042)
    row = store.get_decision(decision_id)
    assert row["arm"] == ARM
    assert row["cost"] == pytest.approx(0.0042)
    assert row["task_type"] == "prose"
    assert row["reward"] is None
    np.testing.assert_allclose(row["context_vector"], vector)
    assert row["context_vector"].shape == (config.CONTEXT_DIM,)


def test_get_decision_returns_none_for_an_unknown_id(store):
    assert store.get_decision(9999) is None


def test_attach_reward_sets_the_reward_and_clears_the_judge_flag(store):
    decision_id = log_one(store, needs_judge=True)
    store.attach_reward(decision_id, 1)
    row = store.get_decision(decision_id)
    assert row["reward"] == 1
    assert row["needs_judge"] == 0


def test_attach_reward_on_an_unknown_id_raises(store):
    with pytest.raises(KeyError):
        store.attach_reward(4242, 1)


def test_pending_judge_returns_only_flagged_unrewarded_rows(store):
    needs = log_one(store, needs_judge=True)
    log_one(store, needs_judge=False)
    already = log_one(store, needs_judge=True, reward=1)
    pending_ids = [row["id"] for row in store.pending_judge(limit=10)]
    assert needs in pending_ids
    assert already not in pending_ids
    assert len(pending_ids) == 1


def test_pending_judge_respects_the_limit(store):
    for _ in range(5):
        log_one(store, needs_judge=True)
    assert len(store.pending_judge(limit=2)) == 2


def test_pending_judge_includes_the_vector_for_a_later_update(store):
    vector = make_vector(0.25)
    log_one(store, needs_judge=True, context_vector=vector)
    row = store.pending_judge(limit=1)[0]
    np.testing.assert_allclose(row["context_vector"], vector)
    assert row["arm"] == ARM
    assert row["query"] == "hello"
    assert row["response_text"] == "hi there"


def test_spend_starts_at_zero_for_an_unknown_conversation(store):
    assert store.get_spent("never-seen") == 0.0


def test_add_spend_accumulates(store):
    store.add_spend("conv-9", 0.001)
    store.add_spend("conv-9", 0.002)
    assert store.get_spent("conv-9") == pytest.approx(0.003)


def test_spend_is_tracked_per_conversation(store):
    store.add_spend("conv-a", 0.005)
    store.add_spend("conv-b", 0.001)
    assert store.get_spent("conv-a") == pytest.approx(0.005)
    assert store.get_spent("conv-b") == pytest.approx(0.001)


def test_state_survives_reopening(tmp_path):
    path = str(tmp_path / "persist.db")
    first = Store(path)
    decision_id = first.log_decision(
        conversation_id="c",
        turn_index=1,
        arm=ARM,
        context_vector=make_vector(0.75),
        retry_depth=0,
        task_type="code",
        cost=0.01,
        reward=1,
        needs_judge=False,
        query="q",
        response_text="r",
    )
    first.add_spend("c", 0.01)
    first.close()

    second = Store(path)
    row = second.get_decision(decision_id)
    assert row["reward"] == 1
    assert row["task_type"] == "code"
    np.testing.assert_allclose(row["context_vector"], make_vector(0.75))
    assert second.get_spent("c") == pytest.approx(0.01)
    second.close()


def test_rejects_a_wrong_length_vector(store):
    with pytest.raises(ValueError, match="context_vector"):
        log_one(store, context_vector=np.zeros(7))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.store'`

- [ ] **Step 3: Write `router/store.py`**

```python
"""SQLite persistence for decisions, rewards, and conversation spend.

The context vector is stored with each decision so a reward arriving later --
from the async judge, seconds or minutes after the response -- can update the
bandit with the vector that arm actually saw at selection time (DESIGN.md
section 6). Without this, delayed rewards silently corrupt learning.
"""

import os
import sqlite3

import numpy as np

from router import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    turn_index      INTEGER NOT NULL,
    arm             TEXT    NOT NULL,
    context_vector  BLOB    NOT NULL,
    retry_depth     INTEGER NOT NULL,
    task_type       TEXT    NOT NULL,
    cost            REAL    NOT NULL,
    reward          INTEGER,
    needs_judge     INTEGER NOT NULL DEFAULT 0,
    query           TEXT,
    response_text   TEXT,
    created_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pending_judge
    ON decisions (needs_judge, reward);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    spent           REAL NOT NULL DEFAULT 0.0
);
"""


class Store:
    def __init__(self, db_path=config.DB_PATH):
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --- decisions

    def log_decision(
        self,
        conversation_id,
        turn_index,
        arm,
        context_vector,
        retry_depth,
        task_type,
        cost,
        reward,
        needs_judge,
        query,
        response_text,
    ):
        vector = np.asarray(context_vector, dtype=np.float64).reshape(-1)
        if vector.shape[0] != config.CONTEXT_DIM:
            raise ValueError(
                f"context_vector has {vector.shape[0]} dims, expected "
                f"{config.CONTEXT_DIM}"
            )
        cursor = self.conn.execute(
            """
            INSERT INTO decisions (
                conversation_id, turn_index, arm, context_vector, retry_depth,
                task_type, cost, reward, needs_judge, query, response_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                int(turn_index),
                arm,
                vector.tobytes(),
                int(retry_depth),
                task_type,
                float(cost),
                reward,
                1 if needs_judge else 0,
                query,
                response_text,
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def _row_to_dict(self, row):
        if row is None:
            return None
        data = dict(row)
        data["context_vector"] = np.frombuffer(
            data["context_vector"], dtype=np.float64
        )
        return data

    def get_decision(self, decision_id):
        row = self.conn.execute(
            "SELECT * FROM decisions WHERE id = ?", (int(decision_id),)
        ).fetchone()
        return self._row_to_dict(row)

    def attach_reward(self, decision_id, reward):
        cursor = self.conn.execute(
            "UPDATE decisions SET reward = ?, needs_judge = 0 WHERE id = ?",
            (int(reward), int(decision_id)),
        )
        self.conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"no decision with id {decision_id}")

    def pending_judge(self, limit):
        rows = self.conn.execute(
            """
            SELECT * FROM decisions
            WHERE needs_judge = 1 AND reward IS NULL
            ORDER BY id ASC LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # --- conversation spend

    def get_spent(self, conversation_id):
        row = self.conn.execute(
            "SELECT spent FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return float(row["spent"]) if row else 0.0

    def add_spend(self, conversation_id, cost):
        self.conn.execute(
            """
            INSERT INTO conversations (conversation_id, spent) VALUES (?, ?)
            ON CONFLICT (conversation_id)
            DO UPDATE SET spent = spent + excluded.spent
            """,
            (conversation_id, float(cost)),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_store.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 7: Bedrock client wrapper

**Files:**
- Create: `router/arms.py`
- Test: `router/tests/test_arms.py`

**Interfaces:**
- Consumes: `config.ARMS`, `config.AWS_PROFILE`, `config.AWS_REGION`.
- Produces:
  - `make_client(profile=config.AWS_PROFILE, region=config.AWS_REGION)` → boto3 `bedrock-runtime` client
  - `bedrock_converse(client, model_id: str, prompt: str, temperature: float | None = None, max_tokens: int = 1024, disable_thinking: bool = False, max_retries: int = 5, sleep_fn=time.sleep) -> tuple[str, dict]` — returns `(text, usage)` where `usage` has `inputTokens` / `outputTokens`
  - `token_cost(model_id: str, usage: dict) -> float`

Carries the two Sonnet 5 quirks from [../README_AWS.md](../README_AWS.md): omit `temperature` unless asked, and pass `thinking: disabled` for the judge or it spends the whole budget on `reasoningContent` and returns no text.

- [ ] **Step 1: Write the failing test**

`router/tests/test_arms.py`:

```python
import pytest

from router import arms, config

ARM = config.ARM_NAMES[0]


class FakeBedrock:
    """Records converse() kwargs and replays scripted responses."""

    def __init__(self, responses=None, errors=None):
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        if self.responses:
            return self.responses.pop(0)
        return make_response("ok")


class ThrottleError(Exception):
    """Stands in for botocore ThrottlingException."""

    def __init__(self):
        super().__init__("ThrottlingException: slow down")
        self.response = {"Error": {"Code": "ThrottlingException"}}


def make_response(text, input_tokens=100, output_tokens=200, blocks=None):
    content = blocks if blocks is not None else [{"text": text}]
    return {
        "output": {"message": {"content": content}},
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


# --- token_cost

def test_token_cost_uses_the_bedrock_usage_keys():
    usage = {"inputTokens": 1_000_000, "outputTokens": 0}
    assert arms.token_cost(ARM, usage) == pytest.approx(config.ARMS[ARM]["input_cost"] * 1e6)


def test_token_cost_sums_input_and_output():
    usage = {"inputTokens": 1000, "outputTokens": 2000}
    expected = (
        1000 * config.ARMS[ARM]["input_cost"] + 2000 * config.ARMS[ARM]["output_cost"]
    )
    assert arms.token_cost(ARM, usage) == pytest.approx(expected)


def test_token_cost_of_missing_usage_is_zero():
    assert arms.token_cost(ARM, {}) == 0.0
    assert arms.token_cost(ARM, None) == 0.0


def test_token_cost_of_an_unknown_arm_raises():
    with pytest.raises(KeyError):
        arms.token_cost("not-a-model", {"inputTokens": 1, "outputTokens": 1})


def test_expensive_arm_costs_more_for_identical_usage():
    usage = {"inputTokens": 1000, "outputTokens": 1000}
    cheap = arms.token_cost(config.ARM_NAMES[0], usage)
    dear = arms.token_cost(config.ARM_NAMES[-1], usage)
    assert dear > cheap


# --- bedrock_converse

def test_returns_text_and_usage():
    client = FakeBedrock([make_response("four", 10, 20)])
    text, usage = arms.bedrock_converse(client, ARM, "what is 2+2")
    assert text == "four"
    assert usage["inputTokens"] == 10
    assert usage["outputTokens"] == 20


def test_prompt_is_sent_as_a_user_message():
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hello")
    call = client.calls[0]
    assert call["modelId"] == ARM
    assert call["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]


def test_temperature_is_omitted_unless_requested():
    """Sonnet 5 rejects temperature in inferenceConfig."""
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hi")
    assert "temperature" not in client.calls[0]["inferenceConfig"]


def test_temperature_is_sent_when_requested():
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hi", temperature=0.0)
    assert client.calls[0]["inferenceConfig"]["temperature"] == 0.0


def test_disable_thinking_sets_the_additional_field():
    """Otherwise Sonnet 5 spends the whole budget on reasoningContent."""
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hi", disable_thinking=True)
    extra = client.calls[0]["additionalModelRequestFields"]
    assert extra == {"thinking": {"type": "disabled"}}


def test_thinking_field_absent_by_default():
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hi")
    assert "additionalModelRequestFields" not in client.calls[0]


def test_text_is_joined_across_all_content_blocks():
    """Never assume content[0] holds the text -- reasoning models put a
    reasoningContent block first."""
    blocks = [
        {"reasoningContent": {"reasoningText": {"text": "hmm"}}},
        {"text": "part one "},
        {"text": "part two"},
    ]
    client = FakeBedrock([make_response("", blocks=blocks)])
    text, _ = arms.bedrock_converse(client, ARM, "hi")
    assert text == "part one part two"


def test_empty_content_yields_an_empty_string():
    client = FakeBedrock([make_response("", blocks=[])])
    text, usage = arms.bedrock_converse(client, ARM, "hi")
    assert text == ""
    assert usage["outputTokens"] == 200


def test_throttling_is_retried_with_backoff():
    sleeps = []
    client = FakeBedrock(
        responses=[make_response("eventually")],
        errors=[ThrottleError(), ThrottleError()],
    )
    text, _ = arms.bedrock_converse(
        client, ARM, "hi", sleep_fn=sleeps.append
    )
    assert text == "eventually"
    assert len(client.calls) == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # exponential


def test_throttling_gives_up_after_max_retries():
    client = FakeBedrock(errors=[ThrottleError() for _ in range(10)])
    with pytest.raises(Exception, match="Throttl"):
        arms.bedrock_converse(
            client, ARM, "hi", max_retries=3, sleep_fn=lambda _: None
        )
    assert len(client.calls) == 3


def test_a_non_throttling_error_is_not_retried():
    class Boom(Exception):
        pass

    client = FakeBedrock(errors=[Boom("bad request")])
    with pytest.raises(Boom):
        arms.bedrock_converse(client, ARM, "hi", sleep_fn=lambda _: None)
    assert len(client.calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_arms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.arms'`

- [ ] **Step 3: Write `router/arms.py`**

```python
"""Bedrock converse wrapper and cost accounting.

Adapted from bedrock_converse in Algorithm/Algo1_AWS.ipynb, including the two
Sonnet 5 quirks documented in README_AWS.md:
  - temperature is only sent when a caller asks for one (Sonnet 5 rejects it)
  - text is joined across all content blocks, never content[0], because a
    reasoning model puts a reasoningContent block first
"""

import time

from router import config


def make_client(profile=config.AWS_PROFILE, region=config.AWS_REGION):
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("bedrock-runtime", region_name=region)


def _is_throttling(error):
    code = ""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
    return "Throttl" in code or "Throttl" in str(error)


def bedrock_converse(
    client,
    model_id,
    prompt,
    temperature=None,
    max_tokens=1024,
    disable_thinking=False,
    max_retries=5,
    sleep_fn=time.sleep,
):
    """Call converse and return (text, usage). Retries throttling only."""
    request = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if temperature is not None:
        request["inferenceConfig"]["temperature"] = temperature
    if disable_thinking:
        request["additionalModelRequestFields"] = {"thinking": {"type": "disabled"}}

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.converse(**request)
            blocks = response.get("output", {}).get("message", {}).get("content", [])
            text = "".join(
                block["text"] for block in blocks if isinstance(block.get("text"), str)
            )
            usage = response.get("usage", {}) or {}
            return text, usage
        except Exception as error:  # noqa: BLE001 - re-raised below
            if not _is_throttling(error):
                raise
            last_error = error
            if attempt < max_retries - 1:
                sleep_fn(2.0 ** attempt)
    raise last_error


def token_cost(model_id, usage):
    """Cost in USD from Bedrock's inputTokens / outputTokens."""
    cfg = config.ARMS[model_id]
    if not usage:
        return 0.0
    return (
        usage.get("inputTokens", 0) * cfg["input_cost"]
        + usage.get("outputTokens", 0) * cfg["output_cost"]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_arms.py -v`
Expected: PASS, 16 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 8: Cascade policy

**Files:**
- Create: `router/policy.py`
- Test: `router/tests/test_policy.py`

**Interfaces:**
- Consumes: `LinUCB`, `ContextBuilder`, `CostEstimator`, `Store`, `verify.check`, `arms.bedrock_converse`, `arms.token_cost`.
- Produces:
  - `class Router(bandit, context_builder, cost_estimator, store, client, converse_fn=arms.bedrock_converse, verify_fn=verify.check)`
  - `.complete(conversation_id: str, query: str, turn_index: int = 0, task_type: str = "prose", expected: str | None = None, schema: dict | None = None, conversation_tokens: int = 0, budget: float | None = None) -> dict` with keys `text`, `arm`, `cascade_depth`, `cost`, `decision_ids`, `verdict`, `status`, `attempts`
  - `.record_reward(decision_id: int, reward: int) -> bool` — `True` if applied, `False` if the decision already had a reward (idempotent); raises `KeyError` on an unknown id

This task carries the plan's one critical correctness rule (§6): each arm is updated with the vector it saw.

- [ ] **Step 1: Write the failing test**

`router/tests/test_policy.py`:

```python
import numpy as np
import pytest

from router import config
from router.bandit import LinUCB
from router.context import ContextBuilder
from router.cost import CostEstimator
from router.policy import Router
from router.store import Store


class FakeEmbedder:
    def encode(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return rng.normal(size=config.EMBED_DIM)


class RecordingBandit(LinUCB):
    """LinUCB that logs every (arm, vector, reward) update."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.updates = []

    def update(self, arm, x, reward):
        self.updates.append((arm, np.array(x, copy=True), reward))
        super().update(arm, x, reward)


def make_router(converse_fn, verify_fn=None, tmp_path=None, forced_arms=None):
    bandit = RecordingBandit(
        arm_names=config.ARM_NAMES,
        dim=config.CONTEXT_DIM,
        alpha=config.ALPHA,
        lambda_reg=config.LAMBDA_REG,
        projection_version=config.PROJECTION_VERSION,
    )
    if forced_arms is not None:
        sequence = list(forced_arms)

        def fake_select(x, candidates, costs, remaining_budget):
            while sequence:
                arm = sequence.pop(0)
                if arm in candidates:
                    return arm, bandit.score(x)
            return None, bandit.score(x)

        bandit.select = fake_select

    store = Store(str(tmp_path / "policy.db"))
    router = Router(
        bandit=bandit,
        context_builder=ContextBuilder(embedder=FakeEmbedder()),
        cost_estimator=CostEstimator(),
        store=store,
        client=object(),
        converse_fn=converse_fn,
        verify_fn=verify_fn if verify_fn is not None else (lambda *a, **k: None),
    )
    return router, bandit, store


def usage(input_tokens=100, output_tokens=100):
    return {"inputTokens": input_tokens, "outputTokens": output_tokens}


def test_single_call_when_the_verifier_passes(tmp_path):
    calls = []

    def converse(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        return "answer", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(calls) == 1
    assert result["cascade_depth"] == 1
    assert result["status"] == "success"
    assert result["verdict"] == 1
    assert bandit.updates[0][2] == 1
    store.close()


def test_escalates_when_the_verifier_fails(tmp_path):
    calls = []
    verdicts = [0, 0, 1]

    def converse(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        return f"answer from {model_id}", usage()

    def verifier(*args, **kwargs):
        return verdicts.pop(0)

    router, bandit, store = make_router(converse, verifier, tmp_path)
    result = router.complete("conv", "q", task_type="code")
    assert len(calls) == 3
    assert len(set(calls)) == 3, "must not retry the same arm"
    assert result["cascade_depth"] == 3
    assert result["status"] == "success"
    assert [u[2] for u in bandit.updates] == [0, 0, 1]
    store.close()


def test_does_not_escalate_on_an_unverifiable_verdict(tmp_path):
    calls = []

    def converse(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        return "prose answer", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: None, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="prose")
    assert len(calls) == 1
    assert result["status"] == "unverified"
    assert result["verdict"] is None
    assert bandit.updates == [], "no reward yet, so no update"
    pending = store.pending_judge(limit=10)
    assert len(pending) == 1
    store.close()


def test_stops_at_max_cascade_depth(tmp_path):
    calls = []

    def converse(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        return "wrong", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 0, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(calls) == config.MAX_CASCADE
    assert result["status"] == "exhausted"
    assert result["cascade_depth"] == config.MAX_CASCADE
    store.close()


def test_each_arm_is_updated_with_the_vector_it_saw(tmp_path):
    """The DESIGN.md section 6 rule. Arm 1 saw retry_depth 0 and no failure;
    arm 2 saw retry_depth 1 and arm 1's one-hot. Updating arm 1 with arm 2's
    vector corrupts learning invisibly."""
    verdicts = [0, 1]

    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, lambda *a, **k: verdicts.pop(0), tmp_path
    )
    router.complete("conv", "q", task_type="code")

    assert len(bandit.updates) == 2
    (first_arm, first_vec, _), (second_arm, second_vec, _) = bandit.updates

    # First arm's vector: no prior failure, retry depth 0
    assert first_vec[35] == 0.0
    assert first_vec[36:43].sum() == 0.0
    assert first_vec[47] == 0.0

    # Second arm's vector: failure flag on, first arm's one-hot set, depth 1
    assert second_vec[35] == 1.0
    first_index = config.ARM_NAMES.index(first_arm)
    assert second_vec[36 + first_index] == 1.0
    assert second_vec[47] == pytest.approx(1.0 / config.MAX_CASCADE)
    assert not np.allclose(first_vec, second_vec)
    store.close()


def test_stops_when_the_budget_is_exhausted(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        return "wrong", usage(input_tokens=10 ** 6, output_tokens=10 ** 6)

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 0, tmp_path=tmp_path
    )
    # Every arm's fallback estimate is affordable at 0.05, so one call happens;
    # every arm's actual cost at 1e6/1e6 tokens exceeds 0.05, so the cascade
    # stops after it rather than escalating.
    result = router.complete("conv", "q", task_type="code", budget=0.05)
    assert result["status"] == "budget_exhausted"
    assert result["cascade_depth"] == 1
    assert result["cost"] > 0.05
    store.close()


def test_returns_budget_exhausted_when_nothing_is_affordable(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        raise AssertionError("must not call any model")

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 0, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code", budget=1e-12)
    assert result["status"] == "budget_exhausted"
    assert result["cascade_depth"] == 0
    assert result["text"] == ""
    store.close()


def test_cost_is_accumulated_per_conversation(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    router.complete("conv-x", "q1", task_type="code")
    first = store.get_spent("conv-x")
    router.complete("conv-x", "q2", task_type="code", turn_index=1)
    assert store.get_spent("conv-x") > first
    assert store.get_spent("conv-other") == 0.0
    store.close()


def test_cost_estimator_learns_from_actual_spend(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path,
        forced_arms=[config.ARM_NAMES[0]] * 4,
    )
    arm = config.ARM_NAMES[0]
    before = router.cost_estimator.estimate(arm, 0)
    for _ in range(3):
        router.complete("conv", "q", task_type="code")
    after = router.cost_estimator.estimate(arm, 0)
    assert after != before
    store.close()


def test_every_attempt_is_logged_with_a_decision_id(tmp_path):
    verdicts = [0, 1]

    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, lambda *a, **k: verdicts.pop(0), tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(result["decision_ids"]) == 2
    for decision_id in result["decision_ids"]:
        assert store.get_decision(decision_id) is not None
    store.close()


def test_a_model_error_is_recorded_and_escalates(tmp_path):
    attempts = []

    def converse(client, model_id, prompt, **kwargs):
        attempts.append(model_id)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return "recovered", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(attempts) == 2
    assert result["status"] == "success"
    assert result["text"] == "recovered"
    # The failed arm is penalised
    assert bandit.updates[0][2] == 0
    store.close()


def test_record_reward_updates_the_bandit_with_the_stored_vector(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        return "prose", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: None, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="prose")
    decision_id = result["decision_ids"][0]
    assert bandit.updates == []

    router.record_reward(decision_id, 1)
    assert len(bandit.updates) == 1
    arm, vector, reward = bandit.updates[0]
    stored = store.get_decision(decision_id)
    assert arm == stored["arm"]
    assert reward == 1
    np.testing.assert_allclose(vector, stored["context_vector"])
    assert stored["reward"] == 1
    store.close()


def test_record_reward_is_idempotent(tmp_path):
    """A retried webhook or a double judge pass must not update twice: two
    updates from one observation double-count it and skew that arm."""

    def converse(client, model_id, prompt, **kwargs):
        return "prose", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: None, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="prose")
    decision_id = result["decision_ids"][0]
    assert router.record_reward(decision_id, 1) is True
    assert router.record_reward(decision_id, 1) is False
    assert len(bandit.updates) == 1
    store.close()


def test_record_reward_ignores_an_already_verified_decision(tmp_path):
    """complete() already updated the bandit for a verified turn, so a later
    /v1/reward for it must be a no-op rather than a second update."""

    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(bandit.updates) == 1
    assert router.record_reward(result["decision_ids"][0], 0) is False
    assert len(bandit.updates) == 1
    store.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_policy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.policy'`

- [ ] **Step 3: Write `router/policy.py`**

```python
"""The cascade. DESIGN.md sections 5 and 6.

Per turn: build context, select an arm under budget, call it, verify. A 0
verdict escalates to a different arm; a None verdict commits and defers to the
sampled judge.

The rule that matters: every bandit update uses the context vector that arm saw
at selection time, not the final one. Arm 2 is chosen with the failure flag set
and retry_depth incremented, so its vector differs from arm 1's. Attributing
arm 2's vector to arm 1 corrupts learning in a way nothing surfaces.
"""

from router import arms as arms_module
from router import config, verify


class Router:
    def __init__(
        self,
        bandit,
        context_builder,
        cost_estimator,
        store,
        client,
        converse_fn=arms_module.bedrock_converse,
        verify_fn=verify.check,
    ):
        self.bandit = bandit
        self.context_builder = context_builder
        self.cost_estimator = cost_estimator
        self.store = store
        self.client = client
        self.converse_fn = converse_fn
        self.verify_fn = verify_fn

    def complete(
        self,
        conversation_id,
        query,
        turn_index=0,
        task_type="prose",
        expected=None,
        schema=None,
        conversation_tokens=0,
        budget=None,
    ):
        total_budget = (
            budget if budget is not None else config.DEFAULT_CONVERSATION_BUDGET
        )
        remaining = total_budget - self.store.get_spent(conversation_id)

        failed_arms = []
        tried = []
        attempts = []
        decision_ids = []
        turn_cost = 0.0
        prompt_tokens = max(1, len(query) // 4)

        for depth in range(config.MAX_CASCADE):
            x = self.context_builder.build(
                query=query,
                turn_index=turn_index,
                prompt_tokens=prompt_tokens,
                conversation_tokens=conversation_tokens,
                task_type=task_type,
                failed_arms=failed_arms,
                retry_depth=depth,
            )
            candidates = [a for a in config.ARM_NAMES if a not in tried]
            costs = self.cost_estimator.estimate_all(turn_index)
            arm, _scores = self.bandit.select(x, candidates, costs, remaining)

            if arm is None:
                status = "budget_exhausted" if depth == 0 else "exhausted"
                return {
                    "text": attempts[-1]["text"] if attempts else "",
                    "arm": attempts[-1]["arm"] if attempts else None,
                    "cascade_depth": depth,
                    "cost": turn_cost,
                    "decision_ids": decision_ids,
                    "verdict": attempts[-1]["verdict"] if attempts else None,
                    "status": status,
                    "attempts": attempts,
                }

            tried.append(arm)
            prompt = self._format_prompt(query, attempts)

            try:
                text, usage = self.converse_fn(
                    self.client, arm, prompt, temperature=0.0
                )
                error = None
            except Exception as exc:  # noqa: BLE001 - recorded as a failure
                text, usage, error = "", {}, str(exc)

            cost = arms_module.token_cost(arm, usage)
            turn_cost += cost
            remaining -= cost
            self.store.add_spend(conversation_id, cost)
            self.cost_estimator.update(arm, turn_index, cost)

            verdict = 0 if error else self.verify_fn(
                task_type, text, expected=expected, schema=schema
            )

            # Update with THIS arm's vector, before the next iteration
            # rebuilds a different one.
            if verdict is not None:
                self.bandit.update(arm, x, float(verdict))

            decision_id = self.store.log_decision(
                conversation_id=conversation_id,
                turn_index=turn_index,
                arm=arm,
                context_vector=x,
                retry_depth=depth,
                task_type=task_type,
                cost=cost,
                reward=verdict,
                needs_judge=verdict is None,
                query=query,
                response_text=text,
            )
            decision_ids.append(decision_id)
            attempts.append(
                {
                    "arm": arm,
                    "text": text,
                    "verdict": verdict,
                    "cost": cost,
                    "error": error,
                    "decision_id": decision_id,
                }
            )

            if verdict is None:
                return {
                    "text": text,
                    "arm": arm,
                    "cascade_depth": depth + 1,
                    "cost": turn_cost,
                    "decision_ids": decision_ids,
                    "verdict": None,
                    "status": "unverified",
                    "attempts": attempts,
                }
            if verdict == 1:
                return {
                    "text": text,
                    "arm": arm,
                    "cascade_depth": depth + 1,
                    "cost": turn_cost,
                    "decision_ids": decision_ids,
                    "verdict": 1,
                    "status": "success",
                    "attempts": attempts,
                }

            failed_arms.append(arm)
            if remaining <= config.EPSILON_COST:
                return {
                    "text": text,
                    "arm": arm,
                    "cascade_depth": depth + 1,
                    "cost": turn_cost,
                    "decision_ids": decision_ids,
                    "verdict": 0,
                    "status": "budget_exhausted",
                    "attempts": attempts,
                }

        last = attempts[-1]
        return {
            "text": last["text"],
            "arm": last["arm"],
            "cascade_depth": len(attempts),
            "cost": turn_cost,
            "decision_ids": decision_ids,
            "verdict": 0,
            "status": "exhausted",
            "attempts": attempts,
        }

    def _format_prompt(self, query, attempts):
        """Include prior failed answers, as the notebooks' format_prompt does."""
        prompt = query
        failed = [a for a in attempts if a["verdict"] == 0 and a["text"]]
        if failed:
            prompt += (
                "\n\nNote: the following previous attempts were incorrect. "
                "Provide a different answer:\n"
            )
            for i, attempt in enumerate(failed, start=1):
                prompt += f"- Attempt {i} (by {attempt['arm']}): {attempt['text']}\n"
        return prompt

    def record_reward(self, decision_id, reward):
        """Apply a delayed reward using the vector stored at selection time.

        Idempotent: a decision that already carries a reward is a no-op.
        A retried webhook or a second judge pass would otherwise update the
        bandit twice from one observation, double-counting it. Returns True
        if the reward was applied.
        """
        row = self.store.get_decision(decision_id)
        if row is None:
            raise KeyError(f"no decision with id {decision_id}")
        if row["reward"] is not None:
            return False
        self.bandit.update(row["arm"], row["context_vector"], float(reward))
        self.store.attach_reward(decision_id, int(reward))
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_policy.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 9: FastAPI service

**Files:**
- Create: `router/app.py`
- Test: `router/tests/test_app.py`

**Interfaces:**
- Consumes: `Router`, `LinUCB`, `ContextBuilder`, `CostEstimator`, `Store`, `arms.make_client`, `config.assert_fallbacks_fit_budget`.
- Produces:
  - `acquire_single_worker_lock(path: str) -> IO` — raises `RuntimeError` if another process holds it
  - `build_app(router=None, lock=True) -> FastAPI`
  - `POST /v1/complete` → `{text, arm, cascade_depth, cost, status, decision_ids}`
  - `POST /v1/reward` → `{ok: true}`
  - `GET /healthz` → `{status, arms, pulls}`

- [ ] **Step 1: Write the failing test**

`router/tests/test_app.py`:

```python
import numpy as np
import pytest
from fastapi.testclient import TestClient

from router import config
from router.app import acquire_single_worker_lock, build_app


class StubRouter:
    """Stands in for policy.Router at the HTTP boundary."""

    def __init__(self):
        self.calls = []
        self.rewards = []
        self.result = {
            "text": "the answer",
            "arm": config.ARM_NAMES[0],
            "cascade_depth": 1,
            "cost": 0.00012,
            "decision_ids": [7],
            "verdict": 1,
            "status": "success",
            "attempts": [],
        }
        self.bandit = type(
            "B", (), {"models": {a: {"pulls": 0} for a in config.ARM_NAMES}}
        )()

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def record_reward(self, decision_id, reward):
        self.rewards.append((decision_id, reward))
        return True


@pytest.fixture
def client():
    stub = StubRouter()
    app = build_app(router=stub, lock=False)
    with TestClient(app) as test_client:
        test_client.stub = stub
        yield test_client


def test_healthz_reports_the_arms(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["arms"]) == 7


def test_complete_returns_the_routed_answer(client):
    response = client.post(
        "/v1/complete",
        json={"conversation_id": "c1", "query": "what is 2+2", "task_type": "math"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "the answer"
    assert body["arm"] == config.ARM_NAMES[0]
    assert body["cascade_depth"] == 1
    assert body["cost"] == pytest.approx(0.00012)
    assert body["decision_ids"] == [7]
    assert body["status"] == "success"


def test_complete_passes_fields_through_to_the_router(client):
    client.post(
        "/v1/complete",
        json={
            "conversation_id": "c2",
            "query": "solve it",
            "task_type": "math",
            "expected": "42",
            "turn_index": 3,
            "conversation_tokens": 900,
        },
    )
    call = client.stub.calls[0]
    assert call["conversation_id"] == "c2"
    assert call["query"] == "solve it"
    assert call["task_type"] == "math"
    assert call["expected"] == "42"
    assert call["turn_index"] == 3
    assert call["conversation_tokens"] == 900


def test_task_type_defaults_to_prose(client):
    client.post("/v1/complete", json={"conversation_id": "c", "query": "hi"})
    assert client.stub.calls[0]["task_type"] == "prose"


def test_an_invalid_task_type_is_rejected(client):
    response = client.post(
        "/v1/complete",
        json={"conversation_id": "c", "query": "hi", "task_type": "poetry"},
    )
    assert response.status_code == 422


def test_a_missing_query_is_rejected(client):
    response = client.post("/v1/complete", json={"conversation_id": "c"})
    assert response.status_code == 422


def test_an_empty_query_is_rejected(client):
    response = client.post(
        "/v1/complete", json={"conversation_id": "c", "query": "   "}
    )
    assert response.status_code == 422


def test_the_serving_arm_is_reported_in_a_header(client):
    """Routing is non-deterministic, so callers need to see what served."""
    response = client.post(
        "/v1/complete", json={"conversation_id": "c", "query": "hi"}
    )
    assert response.headers["x-router-arm"] == config.ARM_NAMES[0]
    assert response.headers["x-router-cascade-depth"] == "1"


def test_reward_endpoint_forwards_to_the_router(client):
    response = client.post("/v1/reward", json={"decision_id": 7, "reward": 1})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "applied": True}
    assert client.stub.rewards == [(7, 1)]


def test_a_duplicate_reward_reports_applied_false(client):
    """record_reward is idempotent; callers see which call actually landed."""
    client.stub.record_reward = lambda decision_id, reward: False
    response = client.post("/v1/reward", json={"decision_id": 7, "reward": 1})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "applied": False}


def test_reward_outside_zero_or_one_is_rejected(client):
    response = client.post("/v1/reward", json={"decision_id": 7, "reward": 5})
    assert response.status_code == 422


def test_reward_for_an_unknown_decision_returns_404(client):
    def boom(decision_id, reward):
        raise KeyError("nope")

    client.stub.record_reward = boom
    response = client.post("/v1/reward", json={"decision_id": 999, "reward": 1})
    assert response.status_code == 404


def test_there_is_no_streaming_endpoint(client):
    """Streaming and silent escalation are incompatible (DESIGN.md section 5)."""
    paths = set(build_app(router=StubRouter(), lock=False).openapi()["paths"])
    assert not any("stream" in path for path in paths)


def test_the_lock_blocks_a_second_holder(tmp_path):
    path = str(tmp_path / "router.lock")
    first = acquire_single_worker_lock(path)
    try:
        with pytest.raises(RuntimeError, match="single worker"):
            acquire_single_worker_lock(path)
    finally:
        first.close()


def test_the_lock_can_be_reacquired_after_release(tmp_path):
    path = str(tmp_path / "router.lock")
    first = acquire_single_worker_lock(path)
    first.close()
    second = acquire_single_worker_lock(path)
    second.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.app'`

- [ ] **Step 3: Write `router/app.py`**

```python
"""HTTP surface. One buffered endpoint, single worker only.

DESIGN.md section 3: two workers would each hold their own A/b and snapshot to
the same file, so the last writer silently discards the other's learning. The
service keeps serving and quietly stops learning, which is why this is enforced
with an exclusive file lock at boot rather than left to documentation.
"""

import fcntl
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from router import config


def acquire_single_worker_lock(path=config.LOCK_PATH):
    """Hold an exclusive lock for the process lifetime, or refuse to boot."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(path, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(
            f"another process already holds {path}. This service must run as a "
            f"single worker (uvicorn --workers 1): each worker would keep its "
            f"own bandit state and overwrite the others' snapshot."
        )
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


class CompleteRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    task_type: str = "prose"
    expected: str | None = None
    schema_: dict | None = Field(default=None, alias="schema")
    turn_index: int = 0
    conversation_tokens: int = 0
    budget: float | None = None

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value):
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("task_type")
    @classmethod
    def task_type_known(cls, value):
        if value not in config.TASK_TYPES:
            raise ValueError(f"task_type must be one of {config.TASK_TYPES}")
        return value


class RewardRequest(BaseModel):
    decision_id: int
    reward: int

    @field_validator("reward")
    @classmethod
    def reward_is_binary(cls, value):
        if value not in (0, 1):
            raise ValueError("reward must be 0 or 1")
        return value


def build_default_router():
    from router import arms
    from router.bandit import LinUCB
    from router.context import ContextBuilder
    from router.cost import CostEstimator
    from router.policy import Router
    from router.store import Store

    config.assert_fallbacks_fit_budget()

    bandit = LinUCB(arm_names=config.ARM_NAMES)
    bandit.restore(config.SNAPSHOT_PATH)
    cost_estimator = CostEstimator()
    cost_estimator.restore(config.COST_STATE_PATH)

    return Router(
        bandit=bandit,
        context_builder=ContextBuilder(),
        cost_estimator=cost_estimator,
        store=Store(config.DB_PATH),
        client=arms.make_client(),
    )


def build_app(router=None, lock=True):
    @asynccontextmanager
    async def lifespan(app):
        yield
        # Persist on shutdown. on_event is deprecated in current FastAPI.
        router_obj = app.state.router
        if hasattr(router_obj, "bandit") and hasattr(router_obj.bandit, "snapshot"):
            router_obj.bandit.snapshot(config.SNAPSHOT_PATH)
        if hasattr(router_obj, "cost_estimator"):
            router_obj.cost_estimator.snapshot(config.COST_STATE_PATH)

    app = FastAPI(title="Per-turn LLM router", lifespan=lifespan)
    app.state.lock_handle = acquire_single_worker_lock() if lock else None
    app.state.router = router if router is not None else build_default_router()

    @app.get("/healthz")
    def healthz():
        bandit = app.state.router.bandit
        return {
            "status": "ok",
            "arms": config.ARM_NAMES,
            "pulls": {
                arm: bandit.models[arm]["pulls"] for arm in bandit.models
            },
        }

    @app.post("/v1/complete")
    def complete(request: CompleteRequest, response: Response):
        result = app.state.router.complete(
            conversation_id=request.conversation_id,
            query=request.query,
            turn_index=request.turn_index,
            task_type=request.task_type,
            expected=request.expected,
            schema=request.schema_,
            conversation_tokens=request.conversation_tokens,
            budget=request.budget,
        )
        response.headers["x-router-arm"] = str(result["arm"])
        response.headers["x-router-cascade-depth"] = str(result["cascade_depth"])
        return {
            "text": result["text"],
            "arm": result["arm"],
            "cascade_depth": result["cascade_depth"],
            "cost": result["cost"],
            "status": result["status"],
            "decision_ids": result["decision_ids"],
        }

    @app.post("/v1/reward")
    def reward(request: RewardRequest):
        try:
            applied = app.state.router.record_reward(
                request.decision_id, request.reward
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown decision_id")
        # applied=False means the decision already had a reward; the call is a
        # no-op, not an error, so a retrying caller can stop retrying.
        return {"ok": True, "applied": bool(applied)}

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_app.py -v`
Expected: PASS, 15 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 10: Sampled judge

**Files:**
- Create: `router/judge.py`
- Test: `router/tests/test_judge.py`

**Interfaces:**
- Consumes: `Store.pending_judge`, `Router.record_reward`, `arms.bedrock_converse`, `config.JUDGE_MODEL`, `config.JUDGE_SAMPLE_RATE`.
- Produces:
  - `build_judge_prompt(query: str, response_text: str) -> str`
  - `parse_verdict(text: str) -> int | None`
  - `run_judge_batch(router, client, limit: int = 20, sample_rate: float = config.JUDGE_SAMPLE_RATE, converse_fn=arms.bedrock_converse, rng=None) -> dict` → `{"considered", "judged", "skipped", "ambiguous"}`

- [ ] **Step 1: Write the failing test**

`router/tests/test_judge.py`:

```python
import numpy as np
import pytest

from router import config, judge


class StubStore:
    def __init__(self, rows):
        self._rows = rows

    def pending_judge(self, limit):
        return self._rows[:limit]


class StubRouter:
    def __init__(self, rows):
        self.store = StubStore(rows)
        self.rewards = []

    def record_reward(self, decision_id, reward):
        self.rewards.append((decision_id, reward))
        return True


def make_row(decision_id=1, query="q", response_text="r"):
    return {
        "id": decision_id,
        "arm": config.ARM_NAMES[0],
        "query": query,
        "response_text": response_text,
        "context_vector": np.zeros(config.CONTEXT_DIM),
    }


class AlwaysTrue:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def __call__(self, client, model_id, prompt, **kwargs):
        self.calls.append({"model_id": model_id, "prompt": prompt, **kwargs})
        reply = self.replies.pop(0) if self.replies else "True"
        return reply, {"inputTokens": 10, "outputTokens": 1}


# --- prompt and parsing

def test_prompt_contains_the_query_and_the_response():
    prompt = judge.build_judge_prompt("what is 2+2", "four")
    assert "what is 2+2" in prompt
    assert "four" in prompt
    assert "True" in prompt and "False" in prompt


def test_parse_verdict_reads_true_and_false_case_insensitively():
    assert judge.parse_verdict("True") == 1
    assert judge.parse_verdict("  true\n") == 1
    assert judge.parse_verdict("FALSE") == 0
    assert judge.parse_verdict("false.") == 0


def test_parse_verdict_returns_none_when_ambiguous():
    assert judge.parse_verdict("maybe") is None
    assert judge.parse_verdict("") is None
    assert judge.parse_verdict(None) is None


# --- batch

def test_judged_rows_get_a_reward():
    router = StubRouter([make_row(5)])
    converse = AlwaysTrue(["True"])
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    assert router.rewards == [(5, 1)]
    assert stats["judged"] == 1
    assert stats["considered"] == 1


def test_a_false_verdict_records_reward_zero():
    router = StubRouter([make_row(6)])
    converse = AlwaysTrue(["False"])
    judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    assert router.rewards == [(6, 0)]


def test_an_ambiguous_verdict_records_nothing():
    router = StubRouter([make_row(7)])
    converse = AlwaysTrue(["I am not sure"])
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    assert router.rewards == []
    assert stats["ambiguous"] == 1
    assert stats["judged"] == 0


def test_sample_rate_zero_judges_nothing():
    router = StubRouter([make_row(i) for i in range(10)])
    converse = AlwaysTrue()
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=0.0, converse_fn=converse
    )
    assert router.rewards == []
    assert converse.calls == []
    assert stats["skipped"] == 10


def test_sample_rate_is_applied_with_the_supplied_rng():
    router = StubRouter([make_row(i) for i in range(100)])
    converse = AlwaysTrue()
    stats = judge.run_judge_batch(
        router,
        client=object(),
        limit=100,
        sample_rate=0.5,
        converse_fn=converse,
        rng=np.random.default_rng(0),
    )
    assert 20 < stats["judged"] < 80
    assert stats["judged"] + stats["skipped"] == 100


def test_the_judge_uses_sonnet_with_thinking_disabled():
    """Sonnet 5 otherwise spends the whole token budget on reasoningContent
    and returns no text at all (README_AWS.md)."""
    router = StubRouter([make_row(1)])
    converse = AlwaysTrue(["True"])
    judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    call = converse.calls[0]
    assert call["model_id"] == config.JUDGE_MODEL
    assert call["disable_thinking"] is True
    assert "temperature" not in call or call["temperature"] is None


def test_a_judge_error_does_not_abort_the_batch():
    calls = []

    def flaky(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        if len(calls) == 1:
            raise RuntimeError("judge unavailable")
        return "True", {"inputTokens": 1, "outputTokens": 1}

    router = StubRouter([make_row(1), make_row(2)])
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=flaky
    )
    assert router.rewards == [(2, 1)]
    assert stats["errors"] == 1


def test_an_empty_queue_is_a_no_op():
    router = StubRouter([])
    converse = AlwaysTrue()
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    assert stats == {
        "considered": 0,
        "judged": 0,
        "skipped": 0,
        "ambiguous": 0,
        "errors": 0,
    }
    assert converse.calls == []


def test_the_default_sample_rate_comes_from_config():
    assert config.JUDGE_SAMPLE_RATE == 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `router/.venv/bin/pytest router/tests/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.judge'`

- [ ] **Step 3: Write `router/judge.py`**

```python
"""Sampled LLM judge supplying delayed reward for unverifiable turns.

DESIGN.md section 6. Runs out of the request path, so latency does not matter.
Rewards land through Router.record_reward, which uses the context vector stored
at selection time -- a reward arriving minutes later is therefore identical in
effect to an immediate one. Bandits are order-insensitive.
"""

import numpy as np

from router import arms as arms_module
from router import config

_PROMPT = """You are grading an AI assistant's response.

User request:
{query}

Assistant response:
{response}

Did the response adequately and correctly answer the request? Respond with only
the word 'True' or 'False'."""


def build_judge_prompt(query, response_text):
    return _PROMPT.format(query=query or "", response=response_text or "")


def parse_verdict(text):
    """1, 0, or None when the judge did not answer usably."""
    if not text:
        return None
    lowered = text.strip().lower()
    if "true" in lowered:
        return 1
    if "false" in lowered:
        return 0
    return None


def run_judge_batch(
    router,
    client,
    limit=20,
    sample_rate=config.JUDGE_SAMPLE_RATE,
    converse_fn=arms_module.bedrock_converse,
    rng=None,
):
    rng = rng if rng is not None else np.random.default_rng()
    rows = router.store.pending_judge(limit=limit)
    stats = {
        "considered": len(rows),
        "judged": 0,
        "skipped": 0,
        "ambiguous": 0,
        "errors": 0,
    }

    for row in rows:
        if rng.random() >= sample_rate:
            stats["skipped"] += 1
            continue
        prompt = build_judge_prompt(row["query"], row["response_text"])
        try:
            # Sonnet 5: no temperature, thinking disabled, or it returns no text.
            text, _usage = converse_fn(
                client,
                config.JUDGE_MODEL,
                prompt,
                max_tokens=10,
                disable_thinking=True,
            )
        except Exception:  # noqa: BLE001 - one bad row must not kill the batch
            stats["errors"] += 1
            continue
        verdict = parse_verdict(text)
        if verdict is None:
            stats["ambiguous"] += 1
            continue
        router.record_reward(row["id"], verdict)
        stats["judged"] += 1
    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `router/.venv/bin/pytest router/tests/test_judge.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Verify nothing entered git**

Run: `git status --short router/`
Expected: no output.

---

### Task 11: Replay harness, full suite, and opt-in smoke test

**Files:**
- Create: `router/replay.py`, `router/tests/test_replay.py`, `router/tests/test_smoke.py`, `router/README.md`
- Test: `router/tests/test_replay.py`, `router/tests/test_smoke.py`

**Interfaces:**
- Consumes: everything above; `Result/algo1res.json`.
- Produces:
  - `load_replay_records(path: str) -> list[dict]` — each `{"query", "expected", "arm_outcomes": {arm: 0|1}, "recorded_cost"}`
  - `replay(records, router_factory, max_records: int | None = None) -> dict` → `{"n", "success_rate", "total_cost", "avg_cascade_depth", "arm_counts"}`

The replay set is the 500 records in [../Result/algo1res.json](../Result/algo1res.json), which log per-arm scores, the chosen arm, cost, and correctness. It answers "would this router have done better than the cascade did" without spending anything.

Note it only knows outcomes for arms the original cascade actually tried, so any arm without a recorded outcome counts as unknown and is skipped, not assumed wrong. This makes replay a sanity check, not an unbiased estimate — an honest limitation of off-policy replay.

- [ ] **Step 1: Write the failing test**

`router/tests/test_replay.py`:

```python
import json

import pytest

from router import config, replay


def make_record(question, truth, attempts):
    return {
        "question": question,
        "ground_truth_answer": truth,
        "final_status": "Success",
        "total_cost": 0.001,
        "attempts": [
            {
                "step": i + 1,
                "chosen_model": arm,
                "is_correct": bool(correct),
                "chosen_model_cost": {"total_cost": 0.0005},
            }
            for i, (arm, correct) in enumerate(attempts)
        ],
    }


@pytest.fixture
def replay_file(tmp_path):
    path = tmp_path / "records.json"
    records = [
        make_record("q1", "42", [(config.ARM_NAMES[0], 0), (config.ARM_NAMES[1], 1)]),
        make_record("q2", "7", [(config.ARM_NAMES[2], 1)]),
    ]
    path.write_text(json.dumps(records))
    return str(path)


def test_loads_queries_and_per_arm_outcomes(replay_file):
    records = replay.load_replay_records(replay_file)
    assert len(records) == 2
    first = records[0]
    assert first["query"] == "q1"
    assert first["expected"] == "42"
    assert first["arm_outcomes"][config.ARM_NAMES[0]] == 0
    assert first["arm_outcomes"][config.ARM_NAMES[1]] == 1


def test_records_only_contain_arms_that_were_actually_tried(replay_file):
    records = replay.load_replay_records(replay_file)
    assert set(records[1]["arm_outcomes"]) == {config.ARM_NAMES[2]}


def test_the_real_result_file_loads_if_present():
    path = "Result/algo1res.json"
    import os

    if not os.path.exists(path):
        pytest.skip("Result/algo1res.json not present")
    records = replay.load_replay_records(path)
    assert len(records) > 100
    assert all("query" in r and "arm_outcomes" in r for r in records)


def test_replay_reports_summary_metrics(replay_file):
    records = replay.load_replay_records(replay_file)
    summary = replay.replay(records, replay.make_offline_router)
    assert summary["n"] == 2
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert summary["total_cost"] >= 0.0
    assert summary["avg_cascade_depth"] >= 1.0
    assert set(summary["arm_counts"]) <= set(config.ARM_NAMES)


def test_replay_honours_max_records(replay_file):
    records = replay.load_replay_records(replay_file)
    summary = replay.replay(records, replay.make_offline_router, max_records=1)
    assert summary["n"] == 1


def test_an_arm_with_no_recorded_outcome_is_skipped_not_failed(tmp_path):
    """Off-policy limitation, stated explicitly: unknown != wrong.

    One record whose only recorded arm is the LAST one by price. The bandit's
    cost penalty makes it pick a cheaper arm first, which has no recorded
    outcome, so the step must count as unknown and the record must not be
    scored as a success.
    """
    path = tmp_path / "sparse.json"
    path.write_text(json.dumps([make_record("q", "1", [(config.ARM_NAMES[-1], 1)])]))
    records = replay.load_replay_records(str(path))
    summary = replay.replay(records, replay.make_offline_router)
    assert summary["unknown_outcomes"] == 1
    assert summary["success_rate"] == 0.0
    assert summary["total_cost"] == 0.0
```

`router/tests/test_smoke.py`:

```python
"""The only test that touches AWS or downloads a model. Opt in with:
    ROUTER_SMOKE=1 router/.venv/bin/pytest router/tests/test_smoke.py -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ROUTER_SMOKE") != "1",
    reason="set ROUTER_SMOKE=1 to run tests that call AWS and load the embedder",
)


def test_the_real_embedder_produces_a_49_dim_vector():
    from router import config
    from router.context import ContextBuilder

    builder = ContextBuilder()
    x = builder.build(
        query="what is the capital of France",
        turn_index=0,
        prompt_tokens=8,
        conversation_tokens=8,
        task_type="prose",
        failed_arms=[],
        retry_depth=0,
    )
    assert x.shape == (config.CONTEXT_DIM,)
    assert x[48] == 1.0


def test_the_cheapest_arm_answers_over_bedrock():
    from router import arms, config

    client = arms.make_client()
    text, usage = arms.bedrock_converse(
        client,
        config.ARM_NAMES[0],
        "Reply with exactly the word: pong",
        temperature=0.0,
        max_tokens=10,
    )
    assert isinstance(text, str) and text.strip()
    assert usage["inputTokens"] > 0
    assert arms.token_cost(config.ARM_NAMES[0], usage) > 0


def test_the_judge_returns_a_parseable_verdict():
    from router import arms, judge

    client = arms.make_client()
    prompt = judge.build_judge_prompt("what is 2+2", "The answer is 4.")
    text, _ = arms.bedrock_converse(
        client, judge.config.JUDGE_MODEL, prompt, max_tokens=10, disable_thinking=True
    )
    assert judge.parse_verdict(text) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `router/.venv/bin/pytest router/tests/test_replay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'router.replay'`

- [ ] **Step 3: Write `router/replay.py`**

```python
"""Offline replay over the notebooks' logged results.

Result/algo1res.json holds 500 questions with the arm chosen at each cascade
step, its cost, and whether it was correct. Replaying it exercises the router
end to end with zero spend.

Limitation, stated rather than hidden: outcomes exist only for arms the original
cascade actually tried. If the router picks an arm with no recorded outcome the
step is skipped as unknown, not scored as wrong. That makes this a sanity check
on wiring and cost, not an unbiased accuracy estimate.
"""

import json

from router import config
from router.bandit import LinUCB
from router.context import ContextBuilder
from router.cost import CostEstimator


class _HashEmbedder:
    """Deterministic offline embedder, no model download.

    hashlib rather than the builtin hash(): PYTHONHASHSEED randomises str
    hashing per process, so builtin hash() would make replay results differ
    between runs.
    """

    def encode(self, text):
        import hashlib

        import numpy as np

        digest = hashlib.blake2b((text or "").encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        return rng.normal(size=config.EMBED_DIM)


def load_replay_records(path):
    with open(path) as handle:
        raw = json.load(handle)

    records = []
    for item in raw:
        outcomes = {}
        recorded_cost = 0.0
        for attempt in item.get("attempts", []):
            arm = attempt.get("chosen_model")
            if arm is None:
                continue
            outcomes[arm] = 1 if attempt.get("is_correct") else 0
            recorded_cost += attempt.get("chosen_model_cost", {}).get("total_cost", 0.0)
        records.append(
            {
                "query": item.get("question", ""),
                "expected": item.get("ground_truth_answer"),
                "arm_outcomes": outcomes,
                "recorded_cost": recorded_cost,
            }
        )
    return records


def make_offline_router():
    """Bandit, context builder, and cost estimator with no network access."""
    return (
        LinUCB(arm_names=config.ARM_NAMES),
        ContextBuilder(embedder=_HashEmbedder()),
        CostEstimator(),
    )


def replay(records, router_factory, max_records=None):
    bandit, context_builder, cost_estimator = router_factory()
    subset = records[:max_records] if max_records else records

    successes = 0
    total_cost = 0.0
    depths = []
    arm_counts = {}
    unknown = 0

    for record in subset:
        tried = []
        failed = []
        depth = 0
        solved = False

        for step in range(config.MAX_CASCADE):
            x = context_builder.build(
                query=record["query"],
                turn_index=0,
                prompt_tokens=max(1, len(record["query"]) // 4),
                conversation_tokens=max(1, len(record["query"]) // 4),
                task_type="math",
                failed_arms=failed,
                retry_depth=step,
            )
            candidates = [a for a in config.ARM_NAMES if a not in tried]
            costs = cost_estimator.estimate_all(0)
            arm, _ = bandit.select(
                x, candidates, costs, config.DEFAULT_CONVERSATION_BUDGET
            )
            if arm is None:
                break

            tried.append(arm)
            depth += 1
            arm_counts[arm] = arm_counts.get(arm, 0) + 1

            outcome = record["arm_outcomes"].get(arm)
            if outcome is None:
                unknown += 1
                break

            step_cost = record["recorded_cost"] / max(
                1, len(record["arm_outcomes"])
            )
            total_cost += step_cost
            cost_estimator.update(arm, 0, step_cost)
            bandit.update(arm, x, float(outcome))

            if outcome == 1:
                solved = True
                break
            failed.append(arm)

        if solved:
            successes += 1
        depths.append(max(depth, 1))

    n = len(subset)
    return {
        "n": n,
        "success_rate": successes / n if n else 0.0,
        "total_cost": total_cost,
        "avg_cascade_depth": sum(depths) / len(depths) if depths else 0.0,
        "arm_counts": arm_counts,
        "unknown_outcomes": unknown,
    }


if __name__ == "__main__":
    records = load_replay_records("Result/algo1res.json")
    summary = replay(records, make_offline_router)
    print(json.dumps(summary, indent=2))
```

- [ ] **Step 4: Run the replay tests to verify they pass**

Run: `router/.venv/bin/pytest router/tests/test_replay.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Run the replay harness against the real 500-record file**

Run: `router/.venv/bin/python -m router.replay`
Expected: JSON with `n: 500`, a `success_rate` between 0 and 1, and `unknown_outcomes` reported. A high `unknown_outcomes` count is expected and is the off-policy limitation, not a bug.

- [ ] **Step 6: Run the whole suite**

Run: `router/.venv/bin/pytest router/tests -v`
Expected: PASS across all modules, 150 tests, with `test_smoke.py`'s 3 tests skipped.

Run pytest from the repo root, not from inside `router/`. Both `router/__init__.py` and `router/tests/__init__.py` exist, so pytest walks up to the repo root before inserting a `sys.path` entry, which is what makes `from router import config` resolve.

- [ ] **Step 7: Write `router/README.md`**

```markdown
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

## Run the service

**Single worker only.** Two workers each keep their own bandit state and
overwrite each other's snapshot; the service keeps serving and silently stops
learning. `app.py` holds an exclusive lock to make that impossible.

    router/.venv/bin/uvicorn router.app:build_app --factory --workers 1 --port 8000

Requires AWS profile `258574424891` with Bedrock access and model access granted
for all seven arms plus `us.anthropic.claude-sonnet-5`.

## Endpoints

- `POST /v1/complete` — `{conversation_id, query, task_type, expected?, schema?, turn_index?, conversation_tokens?, budget?}`
- `POST /v1/reward` — `{decision_id, reward}` for delayed feedback
- `GET /healthz` — arms and pull counts

Responses are buffered; there is no streaming endpoint. Streaming and silent
escalation are incompatible (DESIGN.md §5).

## Known gaps

Cold start, prompt-cache affinity, multi-worker state, and auth are all out of
scope — see DESIGN.md §9.
```

- [ ] **Step 8: Verify nothing entered git**

Run: `git status --short router/ && git status --short`
Expected: the first command prints nothing; the second shows only the pre-existing untracked files in `Algorithm/`, plus `README_AWS.md`, `requirements*.txt`, and `test.ipynb` — no `router/` entries.

---

## Self-Review

**Spec coverage.** Every DESIGN.md section maps to a task:

| Spec section | Task |
| :--- | :--- |
| §3 arms table and prices | 1 |
| §3 module boundaries | 1–11 (one module per task) |
| §3 single-worker invariant | 9 (`acquire_single_worker_lock`) |
| §4 49-d layout, random→PCA projection, version guard | 3 (layout), 2 (`restore` guard) |
| §5 decision path and selection rule | 8 (`Router.complete`), 2 (`LinUCB.select`) |
| §5 buffered API, arm in response | 9 |
| §6 verifier table | 5 |
| §6 sampled judge and Sonnet quirks | 10, 7 |
| §6 per-arm context vector rule | 8 (`test_each_arm_is_updated_with_the_vector_it_saw`), 6 (vector persisted) |
| §7 LCB cost estimator, per-turn keying | 4 |
| §7 per-conversation budget | 6 (`add_spend`/`get_spent`), 8 (pruning) |
| §7 fallback-cost trap and boot assertion | 1 (`assert_fallbacks_fit_budget`), 9 (called in `build_default_router`) |
| §8 test list including replay harness | 2–11 |
| §9 out of scope | not implemented, by design |

**Type consistency.** `check()` returns `int | None` in Task 5 and is consumed as `verdict is None` / `verdict == 1` in Task 8. `estimate_all(turn_index) -> dict` in Task 4 feeds `select(..., costs, ...)` in Task 2. `log_decision` keyword names in Task 6 match the call site in Task 8. `store.pending_judge` rows expose `id`, `arm`, `query`, `response_text`, `context_vector` — all used in Task 10. `Router.record_reward` in Task 8 is what Task 9's `/v1/reward` and Task 10's judge both call.

**One deliberate deviation from the skill template:** no task ends with `git commit`, because the user asked for nothing committed and `router/` is git-excluded. Each task ends with a `git status --short router/` check instead, which fails loudly if the exclusion ever stops working.
