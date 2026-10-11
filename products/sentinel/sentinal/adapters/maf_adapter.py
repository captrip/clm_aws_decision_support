"""
MAF Framework Adapter

Adapter for Multi-Agent Framework (MAF) agents.
"""

from typing import Any, Dict
from .base import BaseFrameworkAdapter


class MAFAdapter(BaseFrameworkAdapter):
    """
    Adapter for MAF framework agents.

    Handles:
    - MAF state format
    - Intent understanding agents
    - Action planning agents
    - Any MAF agent with __call__ method
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute MAF agent.

        MAF agents expect state dict with:
        - user_query
        - updated_query
        - user_id
        - etc.
        """
        # Normalize input to MAF format
        maf_input = self.normalize_input(input_data)

        # Execute agent
        if self._is_async:
            result = await self.agent(maf_input)
        else:
            result = self.agent(maf_input)

        # Normalize output
        return self.normalize_output(result)

    def normalize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert universal input to MAF state format.

        Universal format:
        {
            "query": "user query",
            "user_id": "user_123",
            "context": {...}
        }

        MAF format:
        {
            "user_query": "...",
            "updated_query": "...",
            "user_id": "...",
            ...
        }
        """
        maf_state = {}

        # Map universal keys to MAF keys
        if "query" in input_data:
            maf_state["user_query"] = input_data["query"]
            maf_state["updated_query"] = input_data["query"]

        if "user_id" in input_data:
            maf_state["user_id"] = input_data["user_id"]

        # Pass through any MAF-specific keys
        for key, value in input_data.items():
            if key not in ["query"]:
                maf_state[key] = value

        return maf_state

    def extract_quality_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract quality metrics from MAF agent output.

        MAF agents typically return:
        {
            "intent_understanding": "...",
            "confidence": 0.85,
            "action_plan": [...],
            ...
        }
        """
        metrics = {}

        # Extract confidence if available
        if "confidence" in output:
            metrics["confidence"] = float(output["confidence"])
        else:
            metrics["confidence"] = 0.8  # Default

        # Compute quality based on output completeness
        quality = 0.0

        # Check for intent understanding
        if "intent_understanding" in output:
            intent = output["intent_understanding"]
            if intent and len(str(intent).strip()) > 20:
                quality += 0.5

        # Check for action plan
        if "action_plan" in output or "plan" in output:
            plan = output.get("action_plan") or output.get("plan")
            if plan:
                quality += 0.3

        # Check for other content
        if "result" in output or "response" in output:
            quality += 0.2

        metrics["quality_score"] = min(quality, 1.0)

        # Completeness based on expected fields
        expected_fields = ["intent_understanding", "confidence"]
        present_fields = sum(1 for f in expected_fields if f in output)
        metrics["completeness"] = present_fields / len(expected_fields)

        return metrics
