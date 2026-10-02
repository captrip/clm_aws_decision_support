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
