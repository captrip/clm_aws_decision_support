"""Provider interfaces and logical-model routing."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Protocol

from .models import InvocationResult, ModelRegistry, ModelSpec, ProviderKind


class ProviderConfigurationError(RuntimeError):
    """Raised when a provider cannot be configured for an invocation."""


class ProviderInvocationError(RuntimeError):
    """Raised when a logical-model invocation cannot be completed."""


class ModelProvider(Protocol):
    """Provider adapter for one provider kind."""

    def invoke(
        self,
        model: ModelSpec,
        prompt: str,
        max_tokens: int,
        temperature: float | None,
        additional_fields: Mapping[str, Any] | None = None,
    ) -> InvocationResult: ...


class ProviderRouter:
    """Resolve logical model names and invoke their configured provider."""

    def __init__(
        self, registry: ModelRegistry, providers: Mapping[ProviderKind, ModelProvider]
    ) -> None:
        self._registry = registry
        self._providers = dict(providers)

    def invoke(
        self,
        logical_name: str,
        prompt: str,
        max_tokens: int,
        temperature: float | None,
        additional_fields: Mapping[str, Any] | None = None,
    ) -> InvocationResult:
        try:
            model = self._registry[logical_name]
        except KeyError as error:
            raise ProviderInvocationError(f"unknown logical model: {logical_name}") from error

        provider = self._providers.get(model.provider)
        if provider is None:
            raise ProviderConfigurationError(
                f"no adapter configured for provider {model.provider.value!r} "
                f"and logical model {logical_name!r}"
            )

        try:
            result = provider.invoke(
                model, prompt, max_tokens, temperature, additional_fields=additional_fields
            )
            if not isinstance(result, InvocationResult):
                raise TypeError("provider returned an invalid invocation result")
        except Exception as error:
            raise ProviderInvocationError(
                f"invocation failed for logical model {logical_name!r} "
                f"via provider {model.provider.value!r}"
            ) from error

        return replace(result, logical_model=logical_name)
