from __future__ import annotations
from dataclasses import dataclass

from .features import default_feature_fn
from .grader import calculate_cost
from .models import ModelRegistry, Usage
from .policies.base import DecisionContext, InvalidSelectionError, Selection, SelectionPolicy, required_eligible
from .policies.linucb import LinUCBState
from .providers import ProviderInvocationError, ProviderRouter


@dataclass(frozen=True, slots=True)
class AskResult:
    model: str
    provider: str
    answer: str
    cost: float
    usage: Usage
    selection: Selection
    feature: object
    retried_models: tuple[str,...]

def ask(
        question: str,
        policy: SelectionPolicy,
        router: ProviderRouter,
        registry: ModelRegistry,
        *,
        budget: float | None = None,
        linucb: LinUCBState | None = None,
        max_attempts: int = 3,
        max_token: int = 256,
        temprature: float | None = 0.0,
        feature_fn = default_feature_fn,
) -> AskResult:

    tried: list[str] = []
    last_error: str | None = None

    for step in range(max_attempts):
        eligible = tuple(m for m in registry if m not in tried)
        if not eligible:
            raise InvalidSelectionError(
                f"every model failed with an infrastructure error: {last_error}"
            )

        estimated_costs = {
            m: (linucb.estimate_cost(m) if linucb is not None else 0.0) for m in eligible
        }
        affordable = (
            eligible
            if budget is None
            else tuple(model for model in eligible if estimated_costs[model] <= budget)
        )
        if not affordable:
            raise InvalidSelectionError(
                f"every eligible model's estimated cost exceeds the budget "
                f"{budget}: {estimated_costs!r}"
            )

        features = feature_fn(question, step, len(tried))
        context = DecisionContext(
            question=question,
            step=step,
            failed_models=tuple(tried),
            eligible_models=affordable,
            budget=budget,
            estimated_costs=estimated_costs,
            features=features
        )

        selection = policy.select(context)
        model = selection.model
        required_eligible(model=model, context=context, policy=selection.policy)

        try:
            result = router.invoke(model, question, max_tokens=max_token, temperature=temprature)
        except ProviderInvocationError as error:
            tried.append(model)
            last_error = str(error)
            continue

        model_spec = registry[model]
        cost = calculate_cost(model_spec,result.usage)
        return AskResult(
            model=model,
            provider=model_spec.provider.value,
            answer=result.text,
            cost=cost,
            usage=result.usage,
            selection=selection,
            feature=features,
            retried_models=tuple(tried)
        )

    raise InvalidSelectionError(
        f"every model failed with an infrastructure error: {last_error}"
    )
    