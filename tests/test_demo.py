from __future__ import annotations

import numpy as np
import pytest

from clm_aws_decision_support.demo import (
    SimulatedCLMClient,
    SimulatedGrader,
    SimulatedModelProvider,
    build_demo_registry,
    demo_accuracy,
    seed_demo_deployments,
    visible_answer,
)
from clm_aws_decision_support.ask import ask
from clm_aws_decision_support.cascade import Cascade, Question
from clm_aws_decision_support.features import DEFAULT_FEATURE_DIM
from clm_aws_decision_support.models import ProviderKind
from clm_aws_decision_support.policies.base import DecisionContext
from clm_aws_decision_support.policies.clm import CLMPolicy
from clm_aws_decision_support.policies.linucb import LinUCBPolicy, LinUCBState
from clm_aws_decision_support.providers import ProviderRouter
from clm_aws_decision_support.store import Store


def test_demo_registry_and_seeded_deployments_include_provider_identifiers(
    tmp_path,
) -> None:
    registry = build_demo_registry()
    store = Store(str(tmp_path / "deployments.db"))

    seed_demo_deployments(store)

    deployments = {row["logical_name"]: row for row in store.list_deployments()}
    assert len(registry) == len(deployments) == 5
    assert registry["gpt_4o_mini"].provider is ProviderKind.AZURE_OPENAI
    assert registry["gpt_4o_mini"].deployment == "gpt-4o-mini"
    assert registry["gpt_4o_mini"].model_id is None
    assert deployments["gpt_4o"]["deployment"] == "gpt-4o"
    assert deployments["gpt_4o"]["sample_accuracy"] == demo_accuracy()["gpt_4o"]


def test_store_supports_api_keys_and_decision_feedback(tmp_path) -> None:
    store = Store(str(tmp_path / "workspace.db"))
    key, secret = store.create_api_key("user-123", "test key")

    assert store.verify_api_key(secret)["id"] == key["id"]
    assert store.list_api_keys("user-123")[0]["label"] == "test key"

    store.log_decision(
        "decision-123",
        "question",
        "nova_micro",
        "linucb",
        np.array([1.0, 2.0]),
        0.005,
    )
    decision = store.attach_reward("decision-123", 1.0)

    assert decision is not None
    assert decision["features"].tolist() == [1.0, 2.0]
    assert store.attach_reward("decision-123", 0.0) is None


def test_simulated_provider_and_grader_return_usable_demo_answer() -> None:
    registry = build_demo_registry()
    provider = SimulatedModelProvider(
        accuracy={"nova_micro": 1.0},
        min_latency_s=0,
        max_latency_s=0,
    )

    result = provider.invoke(registry["nova_micro"], "solve this", 32, 0.0)

    assert result.provider is ProviderKind.BEDROCK
    assert result.usage.input_tokens > 0
    assert SimulatedGrader().grade(result.text, "not consulted in simulation")
    assert visible_answer(result.text).startswith("[simulated by nova_micro]")


def test_simulated_clm_matches_policy_response_and_hybrid_criteria() -> None:
    client = SimulatedCLMClient(
        quality={"preferred": 0.9, "other": 0.1},
        seed=1,
    )
    context = DecisionContext(
        question="choose a model",
        step=0,
        failed_models=(),
        eligible_models=("preferred", "other"),
        budget=0.01,
        estimated_costs={"preferred": 0.0, "other": 0.0},
    )

    selection = CLMPolicy(client=client).select(context)

    assert selection.model in context.eligible_models
    assert set(selection.scores) == set(context.eligible_models)
    assert sum(score["probability"] for score in selection.scores.values()) == 1.0
    assert selection.rationale is not None

    response = client.system_one(
        state="hybrid",
        questions={
            "model": {
                "criteria": {
                    "preferred": "LinUCB score = 0.8000estimated cost= $0.010000.",
                    "other": "LinUCB score = 0.4000estimated cost= $0.000000.",
                }
            }
        },
    )
    assert response.answer["model"].probabilities == pytest.approx({
        "preferred": 0.3 / 0.7,
        "other": 0.4 / 0.7,
    })


@pytest.mark.parametrize("budget", [None, 0.01])
def test_ask_routes_a_question_with_or_without_a_budget(budget) -> None:
    registry = build_demo_registry()
    provider = SimulatedModelProvider(
        accuracy={model: 1.0 for model in registry},
        min_latency_s=0,
        max_latency_s=0,
    )
    router = ProviderRouter(
        registry,
        {
            ProviderKind.BEDROCK: provider,
            ProviderKind.AZURE_OPENAI: provider,
        },
    )
    state = LinUCBState(list(registry), dim=DEFAULT_FEATURE_DIM)

    result = ask(
        "What is 2 + 2?",
        LinUCBPolicy(state),
        router,
        registry,
        budget=budget,
        linucb=state,
    )

    assert result.model in registry
    assert result.answer.startswith("[[SIM-OK]]")
    assert result.cost > 0


def test_cascade_runs_without_a_budget_and_reports_a_correct_answer() -> None:
    registry = build_demo_registry()
    provider = SimulatedModelProvider(
        accuracy={model: 1.0 for model in registry},
        min_latency_s=0,
        max_latency_s=0,
    )
    router = ProviderRouter(
        registry,
        {
            ProviderKind.BEDROCK: provider,
            ProviderKind.AZURE_OPENAI: provider,
        },
    )
    state = LinUCBState(list(registry), dim=DEFAULT_FEATURE_DIM)

    result = Cascade(
        LinUCBPolicy(state),
        router,
        SimulatedGrader(),
        registry,
        linucb=state,
    ).run(Question("What is 2 + 2?"), "4")

    assert result.success
    assert len(result.attempts) == 1
    assert result.attempts[0].reward == 1
