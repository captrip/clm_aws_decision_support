from __future__ import annotations
import random
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping
from .models import InvocationResult, ModelRegistry, ModelSpec, ProviderKind, Usage

_DEMO_MODELS: tuple[tuple[str, ProviderKind, str | None, str | None, float, float, float], ...] = (
    ("nova_micro", ProviderKind.BEDROCK,"amazon.nova-micro-v1:0", None, 0.035, 0.14,0.30),
    ("nova_lite", ProviderKind.BEDROCK,"amazon.nova-lite-v1:0", None, 0.06, 0.24,0.45),
    ("gpt_4o_mini", ProviderKind.AZURE_OPENAI, None, "gpt-4o-mini", 0.15, 0.60,0.60),
    ("llama3_70b", ProviderKind.BEDROCK,"us.meta.llama3-3-70b-instruct-v1:0", None, 0.72, 0.72,0.80),
    ("gpt_4o", ProviderKind.AZURE_OPENAI, None, "gpt-4o", 2.50, 10.0,0.92),
)

def build_demo_registry() -> ModelRegistry:
    return ModelRegistry(
        ModelSpec(
            logical_name=name,
            provider=provider,
            model_id=model_id,
            deployment=deployment,
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
        )
        for name, provider, model_id, deployment, input_cost, output_cost, _accuracy in _DEMO_MODELS
    )

def demo_accuracy() -> dict[str, float]:
    return {name: accuracy for name, *_rest, accuracy in _DEMO_MODELS}

def seed_demo_deployments(store: Any) -> None:
    for name, provider, model_id, deployment, input_cost, output_cost, accuracy in _DEMO_MODELS:
        store.add_deployment(
            name,
            provider.value,
            model_id = model_id,
            deployment = deployment,
            input_cost_per_million = input_cost,
            output_cost_per_million = output_cost,
            sample_accuracy = accuracy
        )

_OK_MARKER = "[[SIM-OK]]"
_WRONG_MARKER = "[[SIM-WRONG]]"

@dataclass
class SimulatedModelProvider:

    accuracy: Mapping[str, float]
    seed: int = 0
    min_latency_s: float = 0.05
    max_latency_s: float = 0.35

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def invoke(
            self,
            model: ModelSpec,
            prompt: str,
            max_tokens: int,
            temperature: float | None,
            additional_fields: Mapping[str, Any] | None = None,
    ) -> InvocationResult:
        time.sleep(self._rng.uniform(self.min_latency_s,self.max_latency_s))
        is_correct = self._rng.random() < self.accuracy.get(model.logical_name,0.5)
        marker = _OK_MARKER if is_correct else _WRONG_MARKER
        text = f"{marker} [simulated by {model.logical_name}] This is a placeholder response to: {prompt.strip()}"
        return InvocationResult(
            text=text,
            provider=model.provider,
            usage=Usage(input_tokens=max(1,len(prompt) // 4), output_tokens=max(1,len(text)//4)),
        )

class SimulatedGrader:

    def grade(self, answer: str, expected:str) -> bool:
        return answer.startswith(_OK_MARKER)

def visible_answer(text: str) -> str:
    for marker in (_OK_MARKER, _WRONG_MARKER):
        if text.startswith(marker):
            return text[len(marker) : ].strip()
    return text

_UCB_RE = re.compile(r"LinUCB\s+score\s*=\s*([-+]?\d+(?:\.\d+)?)")
_COST_RE = re.compile(r"estimated cost\s*=\s*\$([-+]?\d+(?:\.\d+)?)")

@dataclass
class SimulatedCLMClient:
    quality: Mapping[str, float]
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def system_one(self,*, state: str, questions: Mapping[str, Any]) -> Any:
        criteria: dict[str, str] = dict(questions["model"]["criteria"])
        weights: dict[str, float] = {}
        for model, description in criteria.items():
            ucb_match = _UCB_RE.search(description)
            cost_match = _COST_RE.search(description)
            if ucb_match or cost_match:
                ucb = float(ucb_match.group(1)) if ucb_match else 0.0
                cost = float(cost_match.group(1)) if cost_match else 0.0
                weights[model] = max(ucb - 50.0 * cost, 0.01)
            else:
                weights[model] = max(self.quality.get(model, 0.5),0.01)

        total = sum(weights.values())
        probabilities = {model: weight / total for model, weight in weights.items()}
        options = list(probabilities)
        chosen = self._rng.choices(
            options, weights=[probabilities[model] for model in options], k=1
        )[0]
        confidence = probabilities[chosen] * len(options)
        model_answer = SimpleNamespace(choice=chosen, probabilities=probabilities)

        return SimpleNamespace(
            answer={"model": model_answer},
            answers = {
                "confidence": SimpleNamespace(score=confidence),
            }
        )