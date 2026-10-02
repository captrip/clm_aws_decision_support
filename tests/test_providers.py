from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from clm_aws_decision_support.models import (
    InvocationResult,
    ModelRegistry,
    ModelSpec,
    ProviderKind,
    Usage,
)
from clm_aws_decision_support.providers import (
    ProviderConfigurationError,
    ProviderInvocationError,
    ProviderRouter,
)


def bedrock_model() -> ModelSpec:
    return ModelSpec(
        logical_name="nova_lite",
        provider=ProviderKind.BEDROCK,
        model_id="amazon.nova-lite-v1:0",
        input_cost_per_million=0.06,
        output_cost_per_million=0.24,
    )


def azure_model() -> ModelSpec:
    return ModelSpec(
        logical_name="gpt_4o_mini",
        provider=ProviderKind.AZURE_OPENAI,
        deployment="gpt-4o-mini-prod",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    )


@dataclass
class FakeProvider:
    result: InvocationResult
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(
        self,
        model: ModelSpec,
        prompt: str,
        max_tokens: int,
        temperature: float | None,
        additional_fields: Mapping[str, Any] | None = None,
    ) -> InvocationResult:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "additional_fields": additional_fields,
            }
        )
        return self.result


def router_with_fakes() -> tuple[ProviderRouter, FakeProvider, FakeProvider]:
    bedrock = FakeProvider(
        InvocationResult("bedrock answer", ProviderKind.BEDROCK, Usage(input_tokens=7))
    )
    azure = FakeProvider(
        InvocationResult("azure answer", ProviderKind.AZURE_OPENAI, Usage(output_tokens=9))
    )
    return (
        ProviderRouter(
            ModelRegistry([bedrock_model(), azure_model()]),
            {ProviderKind.BEDROCK: bedrock, ProviderKind.AZURE_OPENAI: azure},
        ),
        bedrock,
        azure,
    )


def test_router_sends_bedrock_model_to_bedrock_provider() -> None:
    router, bedrock, _ = router_with_fakes()

    result = router.invoke("nova_lite", "solve", 32, 0.0)

    assert result.provider == "bedrock"
    assert result.usage.input_tokens == 7
    assert result.logical_model == "nova_lite"
    assert bedrock.calls[0]["model"] == bedrock_model()


def test_router_sends_azure_deployment_to_azure_openai_provider() -> None:
    router, _, azure = router_with_fakes()

    result = router.invoke("gpt_4o_mini", "solve", 32, 0.0)

    assert result.provider == "azure_openai"
    assert result.usage.output_tokens == 9
    assert result.logical_model == "gpt_4o_mini"
    assert azure.calls[0]["model"] == azure_model()


def test_router_passes_optional_additional_fields_to_selected_provider() -> None:
    router, bedrock, _ = router_with_fakes()
    fields = {"thinking": {"type": "disabled"}}

    router.invoke("nova_lite", "solve", 32, None, additional_fields=fields)

    assert bedrock.calls[0]["additional_fields"] == fields


def test_router_attaches_logical_model_and_provider_to_invocation_errors() -> None:
    router, bedrock, _ = router_with_fakes()
    bedrock.result = None  # type: ignore[assignment]

    with pytest.raises(ProviderInvocationError, match="nova_lite.*bedrock"):
        router.invoke("nova_lite", "solve", 32, 0.0)


def test_bedrock_provider_normalizes_text_blocks_usage_and_request_fields() -> None:
    from clm_aws_decision_support.bedrock import BedrockProvider

    class Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def converse(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {
                "output": {
                    "message": {"content": [{"text": "first"}, {"text": " second"}, {"toolUse": {}}]}
                },
                "usage": {"inputTokens": 7, "outputTokens": 9},
            }

    client = Client()
    result = BedrockProvider(client=client).invoke(
        bedrock_model(),
        "solve",
        32,
        None,
        additional_fields={"thinking": {"type": "disabled"}},
    )

    assert result.text == "first second"
    assert result.usage == Usage(input_tokens=7, output_tokens=9)
    assert client.calls == [
        {
            "modelId": "amazon.nova-lite-v1:0",
            "messages": [{"role": "user", "content": [{"text": "solve"}]}],
            "inferenceConfig": {"maxTokens": 32},
            "additionalModelRequestFields": {"thinking": {"type": "disabled"}},
        }
    ]


def test_bedrock_provider_retries_transient_errors_only() -> None:
    from clm_aws_decision_support.bedrock import BedrockProvider

    class TransientError(Exception):
        response = {"Error": {"Code": "ThrottlingException"}}

    class Client:
        def __init__(self) -> None:
            self.attempts = 0

        def converse(self, **kwargs: Any) -> dict[str, Any]:
            self.attempts += 1
            if self.attempts == 1:
                raise TransientError()
            return {"output": {"message": {"content": [{"text": "ok"}]}}, "usage": {}}

    client = Client()
    delays: list[float] = []

    result = BedrockProvider(client=client, max_attempts=2, sleeper=delays.append).invoke(
        bedrock_model(), "solve", 32, 0.0
    )

    assert result.text == "ok"
    assert client.attempts == 2
    assert delays == [0.25]


def test_azure_provider_uses_injected_client_without_optional_sdk() -> None:
    from clm_aws_decision_support.azure_openai import AzureOpenAIProvider

    class Completions:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="azure answer"))],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=9),
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = AzureOpenAIProvider(client=client).invoke(
        azure_model(),
        "solve",
        32,
        0.0,
        additional_fields={"response_format": {"type": "json_object"}},
    )

    assert result.text == "azure answer"
    assert result.usage == Usage(input_tokens=7, output_tokens=9)
    assert completions.calls == [
        {
            "model": "gpt-4o-mini-prod",
            "messages": [{"role": "user", "content": "solve"}],
            "max_tokens": 32,
            "temperature": 0.0,
            "extra_body": {"response_format": {"type": "json_object"}},
        }
    ]


def test_azure_provider_requires_environment_backed_runtime_configuration() -> None:
    from clm_aws_decision_support.azure_openai import AzureOpenAIProvider

    provider = AzureOpenAIProvider(environment={})

    with pytest.raises(ProviderConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        provider.invoke(azure_model(), "solve", 32, 0.0)
