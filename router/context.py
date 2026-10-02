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
