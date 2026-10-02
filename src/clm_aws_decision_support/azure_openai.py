"""Azure OpenAI adapter with lazy optional dependency loading."""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping

from .models import InvocationResult, ModelSpec, ProviderKind, Usage
from .providers import ProviderConfigurationError


class AzureOpenAIProvider:
    """Invoke Azure OpenAI deployments using environment-backed configuration."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._client = client
        self._environment = environment if environment is not None else os.environ
        self._client_factory = client_factory

    def invoke(
        self,
        model: ModelSpec,
        prompt: str,
        max_tokens: int,
        temperature: float | None,
        additional_fields: Mapping[str, Any] | None = None,
    ) -> InvocationResult:
        if model.provider is not ProviderKind.AZURE_OPENAI:
            raise ValueError("AzureOpenAIProvider requires an azure_openai model")

        request: dict[str, Any] = {
            "model": model.deployment,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if additional_fields is not None:
            request["extra_body"] = dict(additional_fields)

        response = self._get_client().chat.completions.create(**request)
        return InvocationResult(
            text=_content(response),
            provider=ProviderKind.AZURE_OPENAI,
            usage=Usage(
                input_tokens=int(_value(_value(response, "usage", {}), "prompt_tokens", 0) or 0),
                output_tokens=int(
                    _value(_value(response, "usage", {}), "completion_tokens", 0) or 0
                ),
            ),
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        endpoint = self._environment.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ProviderConfigurationError("AZURE_OPENAI_ENDPOINT must be configured")
        api_version = self._environment.get("AZURE_OPENAI_API_VERSION")
        if not api_version:
            raise ProviderConfigurationError("AZURE_OPENAI_API_VERSION must be configured")

        factory = self._client_factory
        if factory is None:
            try:
                from openai import AzureOpenAI
            except ImportError as error:
                raise ProviderConfigurationError(
                    "Azure OpenAI support requires the optional 'openai' package"
                ) from error
            factory = AzureOpenAI

        api_key = self._environment.get("AZURE_OPENAI_API_KEY")
        if api_key:
            self._client = factory(
                azure_endpoint=endpoint, api_version=api_version, api_key=api_key
            )
            return self._client

        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as error:
            raise ProviderConfigurationError(
                "Azure identity authentication requires the optional 'azure-identity' package"
            ) from error
        self._client = factory(
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            ),
        )
        return self._client


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _content(response: Any) -> str:
    choices = _value(response, "choices", [])
    if not choices:
        return ""
    content = _value(_value(choices[0], "message", {}), "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(_value(item, "text", "")) for item in content)
    return "" if content is None else str(content)
