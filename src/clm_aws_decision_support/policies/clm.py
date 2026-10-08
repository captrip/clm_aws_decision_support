from __future__ import annotations
from typing import Any, Mapping, Protocol
from .base import DecisionContext, Selection, required_eligible

class CLMUnavailableError(RuntimeError):
    """Raised when a CLM is unavailable for selection."""

class CLMClientProtocol(Protocol):
    """A protocol for a CLM client."""

    def system_one(self, *, state: str, questions: Mapping[str,Any])-> Any: ...

def default_client(**kwargs: Any)-> CLMClientProtocol:
    try:
        from clm import CLMClient
    except ImportError as error:
        raise CLMUnavailableError(
            "clm mode requires optional clm package"
            "(pip install contrastive-lm) and a running clm-serve"
        ) from error
    return CLMClient(**kwargs)

def default_model_description(model: str) -> str:
    return f"Answer using {model}."

def _model_criteria(
        eligibale_models: tuple[str,...],
        descriptions: Mapping[str,str] | None
) -> dict[str,str]:
    descriptions = descriptions or {}
    return {
        model:descriptions.get(model,default_model_description(model)) for model in eligibale_models
    }

def _state_text(context: DecisionContext) ->str:
    lines = [f"Question: {context.question}",f"Cascade step: {context.step}"]

    if context.failed_models:
        lines.append(f"Already Tried and failed: {','.join(context.failed_models)}")
    if context.budget is not None:
        lines.append(f"Remaining Budget: {context.budget}")

    return "\n".join(lines)

class CLMPolicy:
    name = "clm"

    def __init__(
            self,
            client: CLMClientProtocol | None = None,
            *,
            model_descriptions: Mapping[str,str] | None = None,
            **client_kwargs: Any,
    ) -> None:
        self._client = client if client is not None else default_client(**client_kwargs)
        self._model_descriptions = model_descriptions

    def select(
            self,
            context: DecisionContext,
            *,
            descriptions: Mapping[str,str] | None = None,
            policy_name: str | None = None,
    ) -> Selection:
        name = policy_name or self.name
        criteria = _model_criteria(
            context.eligible_models,
            descriptions if descriptions is not None else self._model_descriptions,
        )
        questions = {
            "model": {
                "instructions": "which model should answer this question?",
                "criteria": criteria,
            },
            "confidence":{
                "instructions": "how confident are you in this choice?",
                "criteria": ["Low","Medium","High"]
            }
        }

        try:
            response = self._client.system_one(state=_state_text(context=context),questions=questions)
        except Exception as error:
            raise CLMUnavailableError(f"CLM server call failed: {error}") from error

        try:
            model_answer = response.answer["model"]
            chosen = model_answer.choice
        except (AttributeError, KeyError) as error:
            raise CLMUnavailableError(f"CLM returned an unusable response: {error}") from error

        required_eligible(chosen, context=context, policy=name)

        scores = {
            model: {"probability": float(prob)}
            for model, prob in dict(getattr(model_answer, "probabilities", {}) or {}).items()
        }
        confidence = getattr(response.answers.get("confidence"), "score", None)
        rationale = f"clm confidence = {confidence}" if confidence is not None else None

        return Selection(models=(chosen,),policy=name, scores=scores, rationale=rationale)
