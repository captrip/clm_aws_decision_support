from __future__ import annotations
import zlib
import numpy as np

DEFAULT_FEATURE_DIM = 16

def default_feature_fn(
        question: str,
        step: int,
        n_failed: int,
        dim: int = DEFAULT_FEATURE_DIM
):
    word_dims = dim-3
    vec = np.zeros(dim, dtype=float)
    for token in question.lower().split():
        idx = zlib.crc32(token.encode("utf-8")) % word_dims
        vec[idx] += 1.0
    norm = np.linalg.norm(vec[:word_dims])
    if norm>0:
        vec[:word_dims] /= norm
    vec[word_dims] = min(step / 10.0, 1.0)
    vec[word_dims + 1] = min(n_failed / 10.0, 1.0)
    vec[word_dims + 2] = 1.0 #bias

    return vec
