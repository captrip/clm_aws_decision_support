from __future__ import annotations
from dataclasses import dataclass

from .features import default_feature_fn
from .grader import AnswerGrader, calculate_cost
from .models import ModelRegistry, Usage
from .policies.base import DecisionContext, InvalidSelectionError, Selection, SelectionPolicy, required_eligible
from .policies.linucb import LinUCBState
from .providers import ProviderInvocationError, ProviderRouter

@dataclass(frozen=True, slots=True)
class Question:
    text:str

@dataclass(frozen=True, slots=True)
class Attempt:
    step: int
    model: str
    policy: str
    selection: Selection
    provider: str | None
    usage: Usage
    cost: float
    answer: str
    reward: int
    error: str | None = None

@dataclass(frozen=True, slots=True)
class QuestionResult:
    attempts: tuple[Attempt, ...]
    success: bool
    total_cost: float

class Cascade:

    def __init__(
            self,
            policy: SelectionPolicy,
            router: ProviderRouter,
            grader: AnswerGrader,
            registry: ModelRegistry,
            *,
            max_attempts: int = 3,
            max_tokens: int = 256,
            temprature: float | None = None,
            linucb: LinUCBState | None = None,
            feature_fn = default_feature_fn,
    ) -> None:
        self._policy = policy
        self._router = router
        self._grader = grader
        self._registry = registry
        self._max_attempts = max_attempts
        self._max_tokens = max_tokens
        self._temprature = temprature
        self._linucb = linucb
        self._feature_fn = feature_fn

    def run(
            self,
            question: Question,
            expected_answer: str,
            budget: float | None = None,
    ) -> QuestionResult:
        attempts: list[Attempt] = []
        failed_models: list[str] = []
        remaining = budget
        total_cost = 0.0

        for step in range(self._max_attempts):
            eligible = tuple(m for m in self._registry if m not in failed_models)
            if not eligible:
                break

            estimated_costs = {model: self._estimate_cost(model) for model in eligible}
            affordable = (
                eligible
                if remaining is None
                else tuple(
                    model for model in eligible if estimated_costs[model] <= remaining
                )
            )
            if not affordable:
                raise InvalidSelectionError(
                    f"every eligible model's estimated cost exceeds the remaining "
                    f"budget {remaining}: {estimated_costs!r}"
                )

            features = self._feature_fn(question.text, step, len(failed_models))
            context = DecisionContext(
                question=question.text,
                step=step,
                failed_models=tuple(failed_models),
                eligible_models=affordable,
                budget=remaining,
                estimated_costs=estimated_costs,
                features=features
            )

            selection = self._policy.select(context=context)
            model = selection.model
            required_eligible(model=model, context=context, policy=selection.policy)

            attempt, reward, cost = self._invoke_and_grade(
                step, model, selection, question, expected_answer
            )
            attempts.append(attempt)
            total_cost += cost
            if remaining is not None:
                remaining -= cost
            if self._linucb is not None:
                self._linucb.update(model, features, reward=float(reward), cost=cost)

            if reward == 1:
                return QuestionResult(
                    attempts=tuple(attempts), success=True, total_cost=total_cost
                )

            failed_models.append(model)
        return QuestionResult(attempts=tuple(attempts), success=False, total_cost=total_cost)

    def _estimate_cost(self, model: str) -> float:
        if self._linucb is not None:
            return self._linucb.estimate_cost(model=model, fallback=0.0)
        return 0.0


    def _invoke_and_grade(
            self,
            step: int,
            model: str,
            selection: Selection,
            question: Question,
            expected_answer: str,
    ) -> tuple[Attempt, int, float]:
        try:
            result = self._router.invoke(model,question.text, self._max_tokens, self._temprature)

        except ProviderInvocationError as error:
            attempt = Attempt(
                step=step,
                model=model,
                policy=selection.policy,
                selection=selection,
                provider=None,
                usage=Usage(),
                cost=0.0,
                answer="",
                reward=0,
                error=str(error)
            )
            return attempt, 0, 0.0

        model_spec = self._registry[model]
        cost = calculate_cost(model_spec, result.usage)
        reward = 1 if self._grader.grade(result.text, expected_answer) else 0
        attempt = Attempt(
                        step=step,
                        model=model,
                        policy=selection.policy,
                        selection=selection,
                        provider=model_spec.provider.value,
                        usage=result.usage,
                        cost=cost,
                        answer=result.text,
                        reward=reward,
                        )
        return attempt, reward, cost