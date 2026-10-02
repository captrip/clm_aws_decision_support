from pathlib import Path

import pytest

from clm_aws_decision_support.config import load_config
from clm_aws_decision_support.models import (
    InvocationOptions,
    ModelRegistry,
    ModelSpec,
    ProviderKind,
    Usage,
)


def write_mixed_provider_config(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        """
models:
  nova_lite:
    provider: bedrock
    model_id: amazon.nova-lite-v1:0
    input_cost_per_million: 0.06
    output_cost_per_million: 0.24
  gpt_4o_mini:
    provider: azure_openai
    deployment: gpt-4o-mini-prod
    input_cost_per_million: 0.15
    output_cost_per_million: 0.60
grader:
  provider: bedrock
  model_id: us.anthropic.claude-sonnet-5
""".strip()
    )
    return path


def bedrock_spec(logical_name: str) -> ModelSpec:
    return ModelSpec(
        logical_name=logical_name,
        provider=ProviderKind.BEDROCK,
        model_id="amazon.nova-lite-v1:0",
        input_cost_per_million=0.06,
        output_cost_per_million=0.24,
    )


def azure_spec(logical_name: str) -> ModelSpec:
    return ModelSpec(
        logical_name=logical_name,
        provider=ProviderKind.AZURE_OPENAI,
        deployment="gpt-4o-mini-prod",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    )


def test_load_config_supports_bedrock_and_azure_openai_models(tmp_path: Path) -> None:
    config = load_config(write_mixed_provider_config(tmp_path))

    assert config.models["nova_lite"].provider is ProviderKind.BEDROCK
    assert config.models["gpt_4o_mini"].provider is ProviderKind.AZURE_OPENAI
    assert config.models["nova_lite"].model_id == "amazon.nova-lite-v1:0"
    assert config.models["gpt_4o_mini"].deployment == "gpt-4o-mini-prod"


def test_model_registry_rejects_duplicate_logical_model_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry([bedrock_spec("same"), azure_spec("same")])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "logical_name": "missing_bedrock_id",
                "provider": ProviderKind.BEDROCK,
                "input_cost_per_million": 0.06,
                "output_cost_per_million": 0.24,
            },
            "model_id",
        ),
        (
            {
                "logical_name": "missing_azure_deployment",
                "provider": ProviderKind.AZURE_OPENAI,
                "input_cost_per_million": 0.15,
                "output_cost_per_million": 0.60,
            },
            "deployment",
        ),
    ],
)
def test_model_spec_requires_its_provider_identifier(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelSpec(**kwargs)


def test_invocation_records_allow_optional_provider_specific_fields() -> None:
    options = InvocationOptions(additional_fields={"thinking": {"type": "disabled"}})
    usage = Usage(input_tokens=7, output_tokens=9)

    assert options.additional_fields == {"thinking": {"type": "disabled"}}
    assert usage.input_tokens == 7
