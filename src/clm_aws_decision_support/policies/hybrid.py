from __future__ import annotations
from typing import Any, Mapping
from .base import DecisionContext, Selection
from .clm import CLMClientProtocol, CLMPolicy, default_model_description
from .linucb import LinUCBState

class HybridPolicy:
    name = "hybrid"

    def __init__(
            self,
            linucb: LinUCBState,
            client: CLMClientProtocol | None = None,
            *,
            model_descriptions: Mapping[str,str] | None = None,
            **client_kwargs: Any,
    ) -> None:
        self.linucb = linucb
        self._model_descriptions = model_descriptions
        self._clm = CLMPolicy(client=client, model_descriptions=model_descriptions, **client_kwargs)

    def select(
            self,
            context: DecisionContext
    ) -> Selection:
        if context.features is None:
            raise ValueError("Hybrid policy Requires Decision Context features")

        ucb_scores = self.linucb.score(context.features)
        base_descriptions = self._model_descriptions or {}
        enriched = {}
        for model in context.eligible_models:
            info = ucb_scores.get(model, {})
            cost = context.estimated_costs.get(model,self.linucb.estimate_cost(model))
            base = base_descriptions.get(model, default_model_description(model))
            enriched[model] = (
                f"{base} LinUCB score = {info.get('ucb',0.0):.4f}"
                f"estimated cost= ${cost:.6f}."
            )

        selection = self._clm.select(context=context, descriptions=enriched, policy_name=self.name)
        return selection