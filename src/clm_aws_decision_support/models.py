"""Provider-neutral configuration and invocation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class ProviderKind(str, Enum):
    BEDROCK = "bedrock"
    AZURE_OPENAI = "azure_openai"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts cannot be negative")


@dataclass(frozen=True, slots=True)
class InvocationOptions:
    """Optional, provider-specific fields carried alongside a neutral call."""

    additional_fields: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.additional_fields is not None:
            object.__setattr__(self, "additional_fields", MappingProxyType(dict(self.additional_fields)))


@dataclass(frozen=True, slots=True)
class InvocationResult:
    text: str
    provider: ProviderKind
    usage: Usage = field(default_factory=Usage)
    logical_model: str | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    logical_name: str
    provider: ProviderKind
    input_cost_per_million: float
    output_cost_per_million: float
    model_id: str | None = None
    deployment: str | None = None

    def __post_init__(self) -> None:
        if not self.logical_name:
            raise ValueError("logical_name is required")
        if self.input_cost_per_million < 0 or self.output_cost_per_million < 0:
            raise ValueError("cost rates cannot be negative")
        if self.provider is ProviderKind.BEDROCK and not self.model_id:
            raise ValueError("model_id is required for bedrock models")
        if self.provider is ProviderKind.AZURE_OPENAI and not self.deployment:
            raise ValueError("deployment is required for azure_openai models")


@dataclass(frozen=True, slots=True)
class GraderSpec:
    provider: ProviderKind
    model_id: str

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("grader model_id is required")


class ModelRegistry(Mapping[str, ModelSpec]):
    """Read-only logical-name lookup with duplicate detection."""

    def __init__(self, models: Iterable[ModelSpec]) -> None:
        entries: dict[str, ModelSpec] = {}
        for model in models:
            if model.logical_name in entries:
                raise ValueError(f"duplicate logical model name: {model.logical_name}")
            entries[model.logical_name] = model
        self._entries = MappingProxyType(entries)

    def __getitem__(self, logical_name: str) -> ModelSpec:
        return self._entries[logical_name]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class AppConfig:
    models: ModelRegistry
    grader: GraderSpec
