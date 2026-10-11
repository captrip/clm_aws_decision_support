from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from clm_aws_decision_support.auth import SupabaseUser, required_supabase_user
from clm_aws_decision_support.store import Store


def test_production_mode_uses_live_provider_adapters() -> None:
    from clm_aws_decision_support import webapp
    from clm_aws_decision_support.azure_openai import AzureOpenAIProvider
    from clm_aws_decision_support.bedrock import BedrockProvider

    providers = webapp._provider_adapters_for_mode("production")

    assert isinstance(providers[webapp.ProviderKind.BEDROCK], BedrockProvider)
    assert isinstance(providers[webapp.ProviderKind.AZURE_OPENAI], AzureOpenAIProvider)


def test_production_provider_configuration_error_is_not_simulated(monkeypatch) -> None:
    from clm_aws_decision_support import webapp
    from clm_aws_decision_support.models import ModelRegistry, ModelSpec, ProviderKind
    from clm_aws_decision_support.providers import (
        ProviderConfigurationError,
        ProviderInvocationError,
        ProviderRouter,
    )

    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    registry = ModelRegistry(
        [
            ModelSpec(
                logical_name="azure-model",
                provider=ProviderKind.AZURE_OPENAI,
                deployment="test-deployment",
                input_cost_per_million=1,
                output_cost_per_million=2,
            )
        ]
    )
    router = ProviderRouter(registry, webapp._provider_adapters_for_mode("production"))

    with pytest.raises(ProviderInvocationError) as error:
        router.invoke("azure-model", "test", 16, None)

    assert isinstance(error.value.__cause__, ProviderConfigurationError)


def test_production_mode_does_not_seed_demo_deployments(tmp_path) -> None:
    from clm_aws_decision_support import webapp

    store = Store(str(tmp_path / "production.db"))
    webapp._initialize_deployments(store, "production")

    assert store.list_deployments() == []


def test_production_mode_rejects_demo_seeded_database(tmp_path) -> None:
    from clm_aws_decision_support import webapp

    store = Store(str(tmp_path / "demo.db"))
    store.set_metadata("registry_source", "demo")

    with pytest.raises(RuntimeError, match="demo-seeded"):
        webapp._initialize_deployments(store, "production")


def test_demo_mode_seeds_demo_deployments_only_when_empty(tmp_path) -> None:
    from clm_aws_decision_support import webapp

    store = Store(str(tmp_path / "demo.db"))
    webapp._initialize_deployments(store, "demo")

    assert len(store.list_deployments()) == 5
    assert store.get_metadata("registry_source") == "demo"


def test_production_policies_and_evaluation_fail_without_configuration(monkeypatch) -> None:
    from fastapi import HTTPException

    from clm_aws_decision_support import webapp
    from clm_aws_decision_support.policies.clm import CLMUnavailableError

    monkeypatch.setattr(webapp, "_RUNTIME_MODE", "production")
    monkeypatch.delenv("CLM_BASE_URL", raising=False)
    monkeypatch.delenv("CLM_GRADER_PROVIDER", raising=False)
    monkeypatch.delenv("CLM_GRADER_MODEL", raising=False)

    with pytest.raises(CLMUnavailableError, match="CLM_BASE_URL"):
        webapp._clm_client()
    with pytest.raises(HTTPException) as error:
        webapp._build_grader()
    assert error.value.status_code == 503
    assert "CLM_GRADER_PROVIDER" in error.value.detail


def test_deployment_accuracy_accepts_fractional_scores() -> None:
    from clm_aws_decision_support.webapp import DeploymentRequest

    deployment = DeploymentRequest(
        logical_name="quality-model",
        provider="azure_openai",
        deployment="test-deployment",
        input_cost_per_million=1,
        output_cost_per_million=2,
        sample_accuracy=0.85,
    )

    assert deployment.sample_accuracy == 0.85


def test_api_key_list_endpoint_returns_list_and_supports_key_creation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CLM_DATA_DIR", str(tmp_path / "app-data"))

    from clm_aws_decision_support import webapp

    monkeypatch.setattr(webapp, "_store", Store(str(tmp_path / "keys.db")))
    monkeypatch.setitem(
        webapp.app.dependency_overrides,
        required_supabase_user,
        lambda: SupabaseUser(id="test-user", email="test@example.com"),
    )

    with TestClient(webapp.app) as client:
        empty_keys = client.get("/api/auth/keys")
        created_key = client.post("/api/auth/keys", json={"label": "test"})
        listed_keys = client.get("/api/auth/keys")

    assert empty_keys.status_code == 200
    assert empty_keys.json() == []
    assert created_key.status_code == 200
    assert created_key.json()["api_key"]
    assert listed_keys.status_code == 200
    assert len(listed_keys.json()) == 1


@pytest.mark.parametrize("product", ["sentinel", "maf"])
def test_product_download_uses_shared_api_key_auth_and_excludes_secrets(
    product, tmp_path, monkeypatch
) -> None:
    from zipfile import ZipFile
    from io import BytesIO

    from clm_aws_decision_support import webapp

    store = Store(str(tmp_path / f"{product}.db"))
    monkeypatch.setattr(webapp, "_store", store)
    _, user_key = store.create_api_key("user-a", label=f"{product}-test")
    _, other_user_key = store.create_api_key("user-b", label="other-user")

    with TestClient(webapp.app) as client:
        unauthorized = client.get(f"/api/products/{product}/download")
        downloaded = client.get(
            f"/api/products/{product}/download",
            headers={"X-API-Key": user_key},
        )
        also_authorized = client.get(
            f"/api/products/{product}/download",
            headers={"X-API-Key": other_user_key},
        )

    assert unauthorized.status_code == 401
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/zip"
    assert also_authorized.status_code == 200
    with ZipFile(BytesIO(downloaded.content)) as package:
        names = package.namelist()
    assert any(name.startswith(f"{product}/") for name in names)
    assert not any(
        ".env" in name or "__pycache__" in name or name.endswith(".pyc")
        for name in names
    )


def test_product_download_rejects_unknown_product(tmp_path, monkeypatch) -> None:
    from clm_aws_decision_support import webapp

    store = Store(str(tmp_path / "products.db"))
    monkeypatch.setattr(webapp, "_store", store)
    _, api_key = store.create_api_key("test-user", label="product-test")

    with TestClient(webapp.app) as client:
        response = client.get(
            "/api/products/unknown/download",
            headers={"X-API-Key": api_key},
        )

    assert response.status_code == 404


def test_mona_ask_tracks_usage_by_api_key_and_exposes_account_summary(
    tmp_path, monkeypatch
) -> None:
    from clm_aws_decision_support import webapp

    store = Store(str(tmp_path / "usage.db"))
    monkeypatch.setattr(webapp, "_store", store)
    _, api_key = store.create_api_key("test-user", label="usage-test")
    monkeypatch.setattr(webapp, "_build_policy", lambda _name: (object(), None))
    monkeypatch.setattr(
        webapp,
        "run_ask",
        lambda **_kwargs: SimpleNamespace(
            model="router-model",
            provider="test-provider",
            answer="Routed answer",
            cost=0.0042,
            usage=SimpleNamespace(input_tokens=14, output_tokens=21),
            selection=SimpleNamespace(rationale="best policy score", scores={}),
            feature=None,
            retried_models=(),
        ),
    )

    with TestClient(webapp.app) as client:
        response = client.post(
            "/api/ask",
            headers={"X-API-Key": api_key},
            json={"question": "Route this", "policy": "linucb", "budget": 0.01},
        )
        usage = client.get("/api/usage", headers={"X-API-Key": api_key})
        unauthorized = client.get("/api/usage")

    assert response.status_code == 200
    assert response.json()["answer"] == "Routed answer"
    assert usage.status_code == 200
    assert usage.json() == {
        "request_count": 1,
        "input_tokens": 14,
        "output_tokens": 21,
        "total_cost": 0.0042,
        "keys": [
            {
                "id": store.list_api_keys("test-user")[0]["id"],
                "label": "usage-test",
                "key_prefix": api_key[:11],
                "request_count": 1,
                "input_tokens": 14,
                "output_tokens": 21,
                "total_cost": 0.0042,
            }
        ],
    }
    assert unauthorized.status_code == 401


def test_maf_demo_uses_account_api_key_for_mona_calls(
    tmp_path, monkeypatch
) -> None:
    from clm_aws_decision_support import webapp

    store = Store(str(tmp_path / "maf-demo.db"))
    monkeypatch.setattr(webapp, "_store", store)
    _, api_key = store.create_api_key("test-user", label="maf-demo")
    client_details: dict[str, str] = {}
    received: dict[str, object] = {}

    class StubMonaAPIError(RuntimeError):
        pass

    class StubMonaClient:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            client_details.update(api_key=api_key, base_url=base_url)

    async def run_demo(**kwargs):
        received.update(kwargs)
        return {
            "answer": "A supervised answer",
            "sentinel": {"status": "completed", "trace": []},
        }

    integration_package = ModuleType("integration")
    integration_package.__path__ = []
    integration_module = ModuleType("integration.conversational_demo")
    integration_module.run_conversational_demo = run_demo
    sentinal_module = ModuleType("sentinal")
    sentinal_module.MonaAPIError = StubMonaAPIError
    sentinal_module.MonaClient = StubMonaClient
    monkeypatch.setitem(sys.modules, "integration", integration_package)
    monkeypatch.setitem(
        sys.modules, "integration.conversational_demo", integration_module
    )
    monkeypatch.setitem(sys.modules, "sentinal", sentinal_module)

    with TestClient(webapp.app) as client:
        missing_key = client.post(
            "/api/maf/demo",
            json={"question": "Plan this task"},
        )
        invalid_key = client.post(
            "/api/maf/demo",
            headers={"X-API-Key": "not-a-valid-key"},
            json={"question": "Plan this task"},
        )
        response = client.post(
            "/api/maf/demo",
            headers={"X-API-Key": api_key},
            json={
                "question": "Plan this task",
                "agent_name": "Planner",
                "instructions": "Be concise.",
                "conversation": [{"role": "assistant", "content": "Ready."}],
                "policy": "hybrid",
                "quality_threshold": 0.8,
            },
        )
        invalid_threshold = client.post(
            "/api/maf/demo",
            headers={"X-API-Key": api_key},
            json={"question": "Plan this task", "quality_threshold": 1.1},
        )

    assert missing_key.status_code == 401
    assert invalid_key.status_code == 401
    assert response.status_code == 200
    assert response.json()["answer"] == "A supervised answer"
    assert client_details == {"api_key": api_key, "base_url": "http://testserver"}
    assert received["agent_name"] == "Planner"
    assert received["policy"] == "hybrid"
    assert received["quality_threshold"] == 0.8
    assert received["conversation"] == [{"role": "assistant", "content": "Ready."}]
    assert invalid_threshold.status_code == 422
