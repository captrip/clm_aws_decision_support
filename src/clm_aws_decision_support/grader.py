from __future__ import annotations

import re
from typing import TYPE_CHECKING
from .models import GraderSpec, ModelSpec, ProviderKind, Usage

if TYPE_CHECKING:
    from .providers import ModelProvider

_PROMPT = """
You are grading a submitted answer against the expected answer.

Expected Answer :
{expected}

Submitted Answer:
{answer}

Does the submitted answer matches the expected answer? Respond with only the word 'true' or 'false'.
"""

_THINKING_DISABLED = {"thinking":{"type":"disabled"}}
_VERDICT = re.compile(r"\b(true|false)\b")
_LEADING_VERDICT = re.compile(r"^\W*(true|false)\b")


def _parse_verdict(text:str) -> bool | None :
    if not text:
        return None
    lowered = text.strip().lower()
    found = set(_VERDICT.findall(lowered))
    if len(found) != 1:
        return None
    leading = _LEADING_VERDICT.match(lowered)
    if leading is None:
        return None
    return leading.group(1) == "true"

def _grader_model(grader: GraderSpec) -> ModelSpec:
    return ModelSpec(
        logical_name="__grader__",
        provider=grader.provider,
        model_id=grader.model_id if grader.provider is ProviderKind.BEDROCK else None,
        deployment=grader.model_id if grader.provider is ProviderKind.AZURE_OPENAI else None,
        input_cost_per_million=0.0,
        output_cost_per_million=0.0,
    )

class AnswerGrader:

    def __init__(
            self,
            provider: "ModelProvider",
            grader: GraderSpec,
            *,
            max_tokens: int = 10
    ) -> None:
        self._provider = provider
        self._model = _grader_model(grader)
        self._max_tokens = max_tokens

    def grade(self, answer: str, expected: str) -> bool:
        if answer.strip() and answer.strip() == expected.strip():
            return True
        prompt = _PROMPT.format(expected = expected, answer = answer)
        result = self._provider.invoke(
            self._model,
            prompt,
            self._max_tokens,
            None,
            additional_fields=_THINKING_DISABLED
        )
        return _parse_verdict(result.text) is True

def calculate_cost(model: ModelSpec, usage: Usage):
    return model.estimated_cost(usage)