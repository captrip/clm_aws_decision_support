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
        projection_seed=config.PROJECTION_SEED,
    ):
        self.arm_names = list(arm_names)
        self.dim = dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.projection_version = projection_version
        self.projection_seed = projection_seed
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
            # The seed defines the projection as much as the label does, so it
            # is versioned too: changing it alone would otherwise load A/b
            # learned in a different feature space under a matching label.
            "__projection_seed__": np.array(self.projection_seed),
            "__dim__": np.array(self.dim),
        }
        for arm, model in self.models.items():
            payload[f"A::{arm}"] = model["A"]
            payload[f"b::{arm}"] = model["b"]
            payload[f"pulls::{arm}"] = np.array(model["pulls"])
        np.savez_compressed(path, **payload)

    def restore(self, path):
        """Load a snapshot. False if absent; raises on a projection mismatch."""
        if not os.path.exists(path):
            return False
        loaded = np.load(path, allow_pickle=False)

        saved_version = str(loaded["__projection_version__"])
        if saved_version != self.projection_version:
            raise ValueError(
                f"snapshot projection version {saved_version!r} != running "
                f"{self.projection_version!r}. The feature space changed meaning, "
                f"so learned A/b are invalid. Delete {path} to start fresh, and "
                f"note that the decisions table holds vectors under the old "
                f"version too -- pending_judge filters them out."
            )
        # A snapshot predating seed versioning cannot be checked, so it cannot
        # be trusted: the seed may have moved underneath it.
        saved_seed = (
            int(loaded["__projection_seed__"])
            if "__projection_seed__" in loaded
            else None
        )
        if saved_seed != self.projection_seed:
            raise ValueError(
                f"snapshot projection seed {saved_seed} != running "
                f"{self.projection_seed}. The seed defines the projection, so "
                f"learned A/b are in a different feature space. Delete {path} "
                f"to start fresh."
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
