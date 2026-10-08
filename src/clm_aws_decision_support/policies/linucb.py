from __future__ import annotations
import json
import math
import os
import numpy as np
from .base import DecisionContext, InvalidSelectionError, Selection

ALPHA = 0.675
LAMBDA_REG = 0.45
EPSILON_COST = 1e-10
MIN_SAMPLE_FOR_LCB = 2
BETA_COST = 1.0

def _empty_cost_bucket() -> dict[str, float]:
    return {"count":0, "total":0.0, "sum_sq": 0.0}

class LinUCBState:
    def __init__(
            self,
            model_names: list[str],
            dim: int,
            alpha: float = ALPHA,
            lambda_reg: float = LAMBDA_REG
    ) -> None:
        self.model_names = list(model_names)
        self.dim = dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self._reward = {
            name: {"A": np.identity(dim) * lambda_reg, "b": np.zeros(dim)}
            for name in self.model_names
        }
        self._cost = {name: _empty_cost_bucket() for name in self.model_names}

    def score(self, features) -> dict[str, dict[str, float]]:

        x = np.asarray(features, dtype=float)
        scores: dict[str, dict[str, float]] = {}
        for name in self.model_names:
            state = self._reward[name]
            try:
                chol = np.linalg.cholesky(state["A"])
                theta = np.linalg.solve(state["A"], state["b"])
                z = np.linalg.solve(chol, x)
                width = self.alpha * float(np.sqrt(np.sum(z**2)))
                mean = float(x.dot(theta))
            except np.linalg.LinAlgError:
                a_inv = np.linalg.inv(state["A"])
                theta = a_inv.dot(state["b"])
                width = self.alpha * float(np.sqrt(x.dot(a_inv).dot(x)))
                mean = float(x.dot(theta))
            scores[name] = {"p": mean, "e": width, "ucb": mean+width}
        return scores

    def estimate_cost(
            self,
            model: str,
            fallback: float =0.0
    ) -> float:
        bucket = self._cost[model]
        if bucket["count"] < MIN_SAMPLE_FOR_LCB:
            return fallback
        n = bucket["count"]
        mean = bucket["total"] / n
        variance = max(0.0, (bucket["sum_sq"]/n)-mean * mean)
        std_error = math.sqrt(variance)/math.sqrt(n)
        return max(EPSILON_COST, mean - BETA_COST * std_error)


    def update(
            self,
            model: str,
            features,
            reward: float,
            cost: float,
    ) -> None:
        x = np.asarray(features, dtype=float)
        state = self._reward[model]
        state["A"] += np.outer(x,x)
        state["b"] += x * float(reward)

        bucket = self._cost[model]
        bucket['count'] +=1
        bucket["total"] += float(cost)
        bucket["sum_sq"] += float(cost) * float(cost)

#------------- dynamic Deployment

    def add_model(
            self,
            model: str,
    ) -> None:
        if model in self._reward:
            return
        self.model_names.append(model)
        self._reward[model] = {"A": np.identity(self.dim) * self.lambda_reg, "b": np.zeros(self.dim)}
        self._cost[model] = _empty_cost_bucket()

    def snapshot(self,path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "__model_names__": np.array(self.model_names),
            "__dim__": np.array(self.dim),
            "__cost__": np.array(json.dumps(self._cost)),
        }
        for name in self.model_names:
            payload[f"A::{name}"] = self._reward[name]["A"]
            payload[f"b::{name}"] = self._reward[name]["b"]
        np.savez_compressed(path, **payload)

    def restore(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        loaded = np.load(path, allow_pickle=False)

        saved_dim = int(loaded["__dim__"])
        if saved_dim != self.dim:
            raise ValueError(f"snapshot dim {saved_dim} != running {self.dim}")

        saved_cost = json.loads(str(loaded["__cost__"]))
        for name in self.model_names:
            if f":A::{name}" in loaded and f"b::{name}" in loaded:
                self._reward[name]["A"] = loaded[f"A::{name}"]
                self._reward[name]["b"] = loaded[f"b::{name}"]
            if name in saved_cost:
                self._cost[name] = saved_cost[name]
        return True

class LinUCBPolicy:

    name = "linucb"
    def __init__(
            self,
            state: LinUCBState
            ) -> None:
        self.state = state

    def select(
            self,
            context: DecisionContext
    ) -> Selection:
        if context.features is None:
            raise ValueError("linucb policy requires Decision context features")

        scores = self.state.score(context.features)
        candidates = [m for m in context.eligible_models if m in scores]
        if not candidates:
            raise InvalidSelectionError(
                "LinUCB: None of the eligible models carry reward State:"
                f"{context.eligible_models}"
            )

        annotated: dict[str,dict[str,float]] = {}
        for model in candidates:
            info = dict(scores[model])
            info["cost_estimate"] = float(context.estimated_costs.get(model, 0.0))
            annotated[model] = info

        max_cost = max((info["cost_estimate"] for info in annotated.values()), default=0.0)
        max_cost = max(max_cost, EPSILON_COST)

        best_model, best_score = None, float("-inf")
        for model, info in annotated.items():
            ratio = (info["cost_estimate"] + EPSILON_COST) / (max_cost+ EPSILON_COST)
            penalty = 1.0 + (1.0 + ratio)
            info["selection_score"] = info['ucb'] / penalty
            if info["selection_score"] > best_score:
                best_score = info["selection_score"]
                best_model = model
        assert best_model is not None
        return Selection(models=(best_model,), policy=self.name, scores=annotated)
    