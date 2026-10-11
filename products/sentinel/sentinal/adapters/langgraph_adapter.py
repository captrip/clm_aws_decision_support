"""
LangGraph Framework Adapter

Adapter for LangGraph agents and graphs.
"""

from typing import Any, Dict
from .base import BaseFrameworkAdapter


class LangGraphAdapter(BaseFrameworkAdapter):
    """
    Adapter for LangGraph framework.

    Handles:
    - LangGraph compiled graphs
    - StateGraph instances
    - Any LangGraph runnable with invoke/ainvoke
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LangGraph graph.

        LangGraph expects state dict and returns updated state.
        """
        # Normalize input
        graph_input = self.normalize_input(input_data)

        # Execute graph
        # LangGraph uses .invoke() or .ainvoke()
        if hasattr(self.agent, 'ainvoke'):
            result = await self.agent.ainvoke(graph_input)
        elif hasattr(self.agent, 'invoke'):
            result = self.agent.invoke(graph_input)
        else:
            # Fallback to direct call
            if self._is_async:
                result = await self.agent(graph_input)
            else:
                result = self.agent(graph_input)

        return self.normalize_output(result)

    def normalize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert universal input to LangGraph format.

        LangGraph typically uses state dicts with specific keys
        defined by the graph's state schema.
        """
        # LangGraph is flexible - usually just pass through
        return input_data

    def extract_quality_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract quality metrics from LangGraph output.

        LangGraph returns the final state dict.
        """
        metrics = {}

        # Check for common quality indicators
        quality = 0.0

        # Check if output has content
        if output:
            quality += 0.3

        # Check for messages (common in LangGraph)
        if "messages" in output:
            messages = output["messages"]
            if messages and len(messages) > 0:
                quality += 0.4

        # Check for final answer or result
        if any(k in output for k in ["answer", "result", "output", "response"]):
            quality += 0.3

        metrics["quality_score"] = min(quality, 1.0)
        metrics["confidence"] = 0.8  # Default
        metrics["completeness"] = quality

        return metrics
