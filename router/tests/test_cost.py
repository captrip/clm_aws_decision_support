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
