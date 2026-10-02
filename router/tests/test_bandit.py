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


def test_restore_rejects_a_mismatched_projection_seed(tmp_path):
    """The seed defines the projection as much as the version label does. With
    the seed unversioned, changing it alone loads A/b learned in a completely
    different feature space while the version string still matches."""
    bandit = make_bandit()
    path = str(tmp_path / "state.npz")
    bandit.snapshot(path)

    other = LinUCB(
        arm_names=["a", "b", "c"],
        dim=4,
        alpha=config.ALPHA,
        lambda_reg=config.LAMBDA_REG,
        projection_version="test-v1",
        projection_seed=999999,
    )
    with pytest.raises(ValueError, match="projection seed"):
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
