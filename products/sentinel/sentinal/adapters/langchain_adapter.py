"""
LangChain Framework Adapter

Adapter for LangChain agents and chains.
"""

from typing import Any, Dict
from .base import BaseFrameworkAdapter


class LangChainAdapter(BaseFrameworkAdapter):
    """
    Adapter for LangChain framework.

    Handles:
    - LangChain agents
    - LangChain chains (LLMChain, SequentialChain, etc.)
    - Any LangChain runnable
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LangChain chain/agent.

        LangChain supports multiple invocation methods.
        """
        # Normalize input
        chain_input = self.normalize_input(input_data)

        # Execute chain/agent
        # LangChain has multiple methods: run, invoke, ainvoke, __call__
        if hasattr(self.agent, 'ainvoke'):
            result = await self.agent.ainvoke(chain_input)
        elif hasattr(self.agent, 'invoke'):
            result = self.agent.invoke(chain_input)
        elif hasattr(self.agent, 'arun'):
            # Old LangChain agents
            result = await self.agent.arun(chain_input)
        elif hasattr(self.agent, 'run'):
            result = self.agent.run(chain_input)
        else:
            # Fallback
            if self._is_async:
                result = await self.agent(chain_input)
            else:
                result = self.agent(chain_input)

        return self.normalize_output(result)

    def normalize_input(self, input_data: Dict[str, Any]) -> Any:
        """
        Convert universal input to LangChain format.

        LangChain chains often expect:
        - Single string input
        - Dict with specific keys
        """
        # If query is provided, extract it
        if "query" in input_data and len(input_data) == 1:
            # Simple chains often take just the query string
            return input_data["query"]

        # Otherwise pass dict
        return input_data

    def normalize_output(self, output: Any) -> Dict[str, Any]:
        """
        Convert LangChain output to universal format.

        LangChain can return:
        - String
        - Dict
        - Complex objects
        """
        if isinstance(output, str):
            return {"result": output, "output": output}

        if isinstance(output, dict):
            return output

        # Complex object - try to extract useful fields
        result_dict = {}

        if hasattr(output, 'content'):
            result_dict["result"] = output.content

        if hasattr(output, 'output'):
            result_dict["output"] = output.output

        if not result_dict:
            result_dict["result"] = str(output)

        return result_dict

    def extract_quality_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract quality metrics from LangChain output.
        """
        metrics = {}

        quality = 0.0

        # Check for output
        if "result" in output or "output" in output:
            content = output.get("result") or output.get("output")
            if content:
                content_str = str(content).strip()
                if len(content_str) > 10:
                    quality += 0.6

                # Longer responses generally indicate better quality
                if len(content_str) > 100:
                    quality += 0.2

        # Check for intermediate steps (agent reasoning)
        if "intermediate_steps" in output:
            quality += 0.2

        metrics["quality_score"] = min(quality, 1.0)
        metrics["confidence"] = 0.75  # Default
        metrics["completeness"] = quality

        return metrics
