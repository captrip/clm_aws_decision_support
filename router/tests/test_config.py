import pathlib

import pytest

import router
from router import config


def test_seven_arms_in_price_order():
    assert len(config.ARM_NAMES) == 7
    assert config.N_ARMS == 7
    assert config.ARM_NAMES[0] == "amazon.nova-micro-v1:0"
    assert config.ARM_NAMES[-1] == "us.meta.llama3-3-70b-instruct-v1:0"
    # Cheapest-first by input price, as in Algo1_AWS.ipynb
    inputs = [config.ARMS[a]["input_cost"] for a in config.ARM_NAMES]
    assert inputs == sorted(inputs)


def test_every_arm_has_both_prices():
    for arm in config.ARM_NAMES:
        assert config.ARMS[arm]["input_cost"] > 0
        assert config.ARMS[arm]["output_cost"] > 0


def test_exact_arm_prices():
    # Guard against transcription errors: a 10x typo in arm prices would corrupt
    # every cost calculation and budget decision downstream. Assert exact values.
    assert config.ARMS["amazon.nova-micro-v1:0"]["input_cost"] == 0.035 / 1e6
    assert config.ARMS["amazon.nova-micro-v1:0"]["output_cost"] == 0.14 / 1e6
    assert config.ARMS["amazon.nova-lite-v1:0"]["input_cost"] == 0.06 / 1e6
    assert config.ARMS["amazon.nova-lite-v1:0"]["output_cost"] == 0.24 / 1e6
    assert config.ARMS["mistral.mistral-7b-instruct-v0:2"]["input_cost"] == 0.15 / 1e6
    assert config.ARMS["mistral.mistral-7b-instruct-v0:2"]["output_cost"] == 0.20 / 1e6
    assert config.ARMS["us.meta.llama4-scout-17b-instruct-v1:0"]["input_cost"] == 0.17 / 1e6
    assert config.ARMS["us.meta.llama4-scout-17b-instruct-v1:0"]["output_cost"] == 0.66 / 1e6
    assert config.ARMS["us.meta.llama3-1-8b-instruct-v1:0"]["input_cost"] == 0.22 / 1e6
    assert config.ARMS["us.meta.llama3-1-8b-instruct-v1:0"]["output_cost"] == 0.22 / 1e6
    assert config.ARMS["us.meta.llama4-maverick-17b-instruct-v1:0"]["input_cost"] == 0.24 / 1e6
    assert config.ARMS["us.meta.llama4-maverick-17b-instruct-v1:0"]["output_cost"] == 0.97 / 1e6
    assert config.ARMS["us.meta.llama3-3-70b-instruct-v1:0"]["input_cost"] == 0.72 / 1e6
    assert config.ARMS["us.meta.llama3-3-70b-instruct-v1:0"]["output_cost"] == 0.72 / 1e6


def test_context_dim_matches_layout():
    # 32 embedding + 1 turn + 1 prompt tok + 1 conv tok + 1 fail flag
    # + 7 last-failed-arm + 4 task type + 1 retry depth + 1 bias
    assert config.PROJ_DIM == 32
    assert config.CONTEXT_DIM == 32 + 1 + 1 + 1 + 1 + 7 + 4 + 1 + 1
    assert config.CONTEXT_DIM == 49


def test_task_types_are_the_four_from_the_spec():
    # Order matters: later tasks index a one-hot vector by position in this list.
    assert config.TASK_TYPES == ["code", "structured", "math", "prose"]


def test_notebook_hyperparameters():
    assert config.ALPHA == 0.675
    assert config.LAMBDA_REG == 0.45
    assert config.BETA_COST == 1.0


def test_fallback_cost_is_per_arm_and_price_ordered():
    cheap = config.fallback_cost("amazon.nova-micro-v1:0")
    dear = config.fallback_cost("us.meta.llama3-3-70b-instruct-v1:0")
    assert 0 < cheap < dear


def test_fallback_costs_fit_the_default_budget():
    # The Algo3 trap: a global fallback 70x the budget means nothing ever fits
    # and every turn exits at zero steps. Must fail loudly instead.
    config.assert_fallbacks_fit_budget()


def test_assert_fallbacks_raises_when_budget_too_small(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_CONVERSATION_BUDGET", 1e-12)
    with pytest.raises(ValueError, match="fallback cost"):
        config.assert_fallbacks_fit_budget()


def test_state_paths_are_absolute_and_under_the_package():
    """Relative paths resolve against the cwd, so booting from elsewhere reads a
    snapshot that is not there, discards every pull of learning without a word,
    and takes out a lock a second worker in another cwd would not see."""
    package_dir = pathlib.Path(router.__file__).resolve().parent
    for path in (
        config.SNAPSHOT_PATH,
        config.COST_STATE_PATH,
        config.DB_PATH,
        config.LOCK_PATH,
    ):
        assert pathlib.Path(path).is_absolute(), path
        assert package_dir in pathlib.Path(path).parents, path
