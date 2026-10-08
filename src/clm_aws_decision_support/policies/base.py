from __future__ import annotations
from dataclasses import dataclass,field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol

if TYPE_CHECKING:
    from numpy.typing import NDArray

class InvalidSelectionError(ValueError):
    """Raised when a selection is invalid."""


@dataclass(frozen=True,slots=True)
class DecisionContext:

    question: str
    step: int
    failed_models: tuple[str,...]
    eligible_models: tuple[str,...]
    budget: float | None
    estimated_costs: Mapping[str,float] = field(default_factory=dict)
    features:"NDArray | None" = None

    def __post_init__(self):
        if not self.eligible_models:
            raise ValueError("No eligible models available.")
        object.__setattr__(self,"estimated_costs",MappingProxyType(self.estimated_costs))

@dataclass(frozen=True,slots=True)
class Selection:

    models: tuple[str,...]
    policy: str
    scores: Mapping[str,Mapping[str,float]] = field(default_factory=dict)
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("No models selected.")
        object.__setattr__(self,"scores",MappingProxyType(dict(self.scores)))

    @property
    def model(self) -> str:
        """Return the first model in the selection."""
        return self.models[0]

class SelectionPolicy(Protocol):
    """A policy for selecting models."""

    def select(self, context: DecisionContext) -> Selection: ...

def required_eligible(model: str, context: DecisionContext,*,policy: str) -> None:
    """Raise InvalidSelectionError if model is not eligible."""
    if model not in context.eligible_models:
        raise InvalidSelectionError(
            f"{policy} selected '{model!r}', which is not eligible for selection in this context."
            f" Eligible models: {context.eligible_models!r}"
        )