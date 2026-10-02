"""YAML configuration loading for model and grader registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import AppConfig, GraderSpec, ModelRegistry, ModelSpec, ProviderKind


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _provider(value: object, description: str) -> ProviderKind:
    try:
        return ProviderKind(str(value))
    except ValueError as error:
        raise ValueError(f"{description} has unsupported provider: {value!r}") from error


def _cost(value: Mapping[str, Any], field_name: str, logical_name: str) -> float:
    if field_name not in value:
        raise ValueError(f"{logical_name} is missing {field_name}")
    return float(value[field_name])


def load_config(path: Path) -> AppConfig:
    """Load model candidates and the configured answer grader from YAML."""
    loaded = yaml.safe_load(path.read_text())
    document = _mapping(loaded, "configuration")
    raw_models = _mapping(document.get("models"), "models")

    models = []
    for logical_name, raw_spec in raw_models.items():
        spec = _mapping(raw_spec, f"model {logical_name!r}")
        provider = _provider(spec.get("provider"), f"model {logical_name!r}")
        models.append(
            ModelSpec(
                logical_name=str(logical_name),
                provider=provider,
                model_id=spec.get("model_id"),
                deployment=spec.get("deployment"),
                input_cost_per_million=_cost(spec, "input_cost_per_million", str(logical_name)),
                output_cost_per_million=_cost(spec, "output_cost_per_million", str(logical_name)),
            )
        )

    raw_grader = _mapping(document.get("grader"), "grader")
    grader = GraderSpec(
        provider=_provider(raw_grader.get("provider"), "grader"),
        model_id=str(raw_grader.get("model_id") or ""),
    )
    return AppConfig(models=ModelRegistry(models), grader=grader)
