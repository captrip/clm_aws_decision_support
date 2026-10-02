"""Offline replay over the notebooks' logged results.

Result/algo1res.json holds 500 questions with the arm chosen at each cascade
step, its cost, and whether it was correct. Replaying it exercises the router
end to end with zero spend.

Two limitations, stated rather than hidden.

First, outcomes exist only for arms the original cascade actually tried. If the
router picks an arm with no recorded outcome the step is skipped as unknown, not
scored as wrong.

Second, that log is the OpenRouter experiment: its model names are
google/gemini-2.0-flash-001, deepseek/deepseek-chat and so on, none of which is
one of this router's seven Bedrock arms. Left unmapped, the name sets are
disjoint and EVERY record scores as unknown -- 500 unknowns, a 0.0 success rate
and 0.0 cost, a harness that exercises nothing. _ARM_ALIASES maps each logged
model onto the Bedrock arm closest to it in capability tier so the records
actually drive the bandit. The correctness and cost therefore come from a
different model than the arm being credited.

Both limitations point the same way: this is a sanity check on wiring, selection
and cost accounting, not an unbiased accuracy estimate. Algorithm/M91_AWS.json
is the only log whose names are genuinely this router's arms, and it holds 5
records -- too few to measure anything.
"""

import json

from router import config
from router.bandit import LinUCB
from router.context import ContextBuilder
from router.cost import CostEstimator


"""Each OpenRouter model in the log, mapped to the Bedrock arm nearest it in
capability tier.

Cheapest-first, matching the price ordering of config.ARM_NAMES. A logged model
with no entry here keeps its own name and will simply never match an arm, which
is the honest outcome rather than a silent mis-credit.
"""
_ARM_ALIASES = {
    "openai/gpt-4.1-nano": "amazon.nova-micro-v1:0",
    "google/gemini-2.0-flash-001": "amazon.nova-lite-v1:0",
    "mistralai/mistral-small-3.1-24b-instruct": "mistral.mistral-7b-instruct-v0:2",
    "microsoft/phi-4": "us.meta.llama3-1-8b-instruct-v1:0",
    "meta-llama/llama-4-maverick": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "deepseek/deepseek-chat": "us.meta.llama3-3-70b-instruct-v1:0",
}


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
            logged = attempt.get("chosen_model")
            if logged is None:
                continue
            arm = _ARM_ALIASES.get(logged, logged)
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


def covered_arms(records):
    """The arms this record set can actually score.

    Only 6 models were logged, against 7 arms, so at least one arm has no
    recorded outcome anywhere. An arm that is never updated keeps its full
    initial optimism forever, so the bandit picks it, finds no outcome, and
    the record is thrown away -- unmeasured, llama4-scout took 404 of the 500
    first picks on its own. Restricting candidates to the covered set is
    standard off-policy rejection sampling: the arms outside it are not
    evaluated rather than silently distorting every step.
    """
    seen = {arm for record in records for arm in record["arm_outcomes"]}
    return [arm for arm in config.ARM_NAMES if arm in seen]


def replay(records, router_factory, max_records=None):
    bandit, context_builder, cost_estimator = router_factory()
    # `is not None`, not truthiness: max_records=0 must mean "none", but 0 is
    # falsy and would silently replay all 500.
    subset = records[:max_records] if max_records is not None else records
    scoreable = covered_arms(subset)

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
            candidates = [a for a in scoreable if a not in tried]
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
        "scoreable_arms": len(scoreable),
    }


if __name__ == "__main__":
    records = load_replay_records("Result/algo1res.json")
    summary = replay(records, make_offline_router)
    print(json.dumps(summary, indent=2))
