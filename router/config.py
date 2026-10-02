"""Constants for the per-turn router. See DESIGN.md sections 3, 4, 7."""

import pathlib

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
# Anchored to this package, not the cwd. Relative paths mean a service started
# from another directory reads a snapshot that is not there and boots cold with
# every pull of learning gone, silently -- and takes out a lock file a second
# worker in a different cwd would never contend with, defeating the
# single-worker invariant (DESIGN.md section 3). Kept as str: Store,
# acquire_single_worker_lock, and np.load/savez all take either.
_STATE_DIR = pathlib.Path(__file__).resolve().parent / "state"
SNAPSHOT_PATH = str(_STATE_DIR / "bandit.npz")
COST_STATE_PATH = str(_STATE_DIR / "cost.npz")
DB_PATH = str(_STATE_DIR / "router.db")
LOCK_PATH = str(_STATE_DIR / "router.lock")


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
