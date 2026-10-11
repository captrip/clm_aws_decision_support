"""
CrewAI Framework Adapter

Adapter for CrewAI agents and crews.
"""

from typing import Any, Dict
from .base import BaseFrameworkAdapter


class CrewAIAdapter(BaseFrameworkAdapter):
    """
    Adapter for CrewAI framework.

    Handles:
    - CrewAI agents
    - CrewAI crews
    - CrewAI tasks
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute CrewAI agent or crew.

        CrewAI has kickoff() method for execution.
        """
        # Normalize input
        crew_input = self.normalize_input(input_data)

        # Execute
        # CrewAI uses .kickoff() method
        if hasattr(self.agent, 'kickoff'):
            result = self.agent.kickoff(crew_input)
        elif hasattr(self.agent, 'execute'):
            result = self.agent.execute(crew_input)
        else:
            # Fallback
            if self._is_async:
                result = await self.agent(crew_input)
            else:
                result = self.agent(crew_input)

        return self.normalize_output(result)

    def normalize_input(self, input_data: Dict[str, Any]) -> Any:
        """
        Convert universal input to CrewAI format.

        CrewAI often expects:
        - inputs dict for crews
        - specific task parameters
        """
        # CrewAI crews take inputs dict
        if "query" in input_data:
            return {"inputs": {"query": input_data["query"]}}

        return input_data

    def normalize_output(self, output: Any) -> Dict[str, Any]:
        """
        Convert CrewAI output to universal format.

        CrewAI returns CrewOutput object or string.
        """
        if isinstance(output, str):
            return {"result": output, "output": output}

        if isinstance(output, dict):
            return output

        # CrewOutput object
        result_dict = {}

        if hasattr(output, 'raw'):
            result_dict["result"] = output.raw

        if hasattr(output, 'json_dict'):
            result_dict["structured_output"] = output.json_dict

        if hasattr(output, 'tasks_output'):
            result_dict["tasks_output"] = output.tasks_output

        if not result_dict:
            result_dict["result"] = str(output)

        return result_dict

    def extract_quality_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract quality metrics from CrewAI output.
        """
        metrics = {}

        quality = 0.0

        # Check for result
        if "result" in output:
            result = output["result"]
            if result and len(str(result).strip()) > 20:
                quality += 0.5

        # Check for structured output (indicates quality)
        if "structured_output" in output or "json_dict" in output:
            quality += 0.3

        # Check for tasks output (crew execution)
        if "tasks_output" in output:
            tasks = output["tasks_output"]
            if tasks and len(tasks) > 0:
                quality += 0.2

        metrics["quality_score"] = min(quality, 1.0)
        metrics["confidence"] = 0.8  # Default
        metrics["completeness"] = quality

        return metrics
