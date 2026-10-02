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
