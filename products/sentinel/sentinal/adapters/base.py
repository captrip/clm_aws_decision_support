"""
Base Adapter for Universal Framework Integration

Defines the interface that all framework adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, Optional
import asyncio


class BaseFrameworkAdapter(ABC):
    """
    Base adapter that any framework must implement.

    This allows Sentinel to supervise agents from:
    - MAF
    - LangGraph
    - LangChain
    - CrewAI
    - Custom frameworks

    The adapter translates between framework-specific APIs
    and Sentinel's universal execution interface.
    """

    def __init__(self, agent: Any, agent_name: str):
        """
        Initialize adapter.

        Args:
            agent: The framework-specific agent instance
            agent_name: Human-readable name for the agent
        """
        self.agent = agent
        self.agent_name = agent_name
        self._is_async = self._detect_async()

    def _detect_async(self) -> bool:
        """Detect if agent uses async/await."""
        if hasattr(self.agent, '__call__'):
            import inspect
            return inspect.iscoroutinefunction(self.agent.__call__)
        return False

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent.

        Args:
            input_data: Framework-agnostic input dictionary

        Returns:
            Framework-agnostic output dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def extract_quality_metrics(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract quality metrics from agent output.

        Args:
            output: Agent's output

        Returns:
            Dictionary of quality metrics:
            {
                "quality_score": 0.0-1.0,
                "confidence": 0.0-1.0,
                "completeness": 0.0-1.0,
            }
        """
        raise NotImplementedError

    def normalize_input(self, input_data: Dict[str, Any]) -> Any:
        """
        Convert universal input to framework-specific format.

        Override this if your framework needs special input format.

        Args:
            input_data: Universal input dict

        Returns:
            Framework-specific input
        """
        return input_data

    def normalize_output(self, output: Any) -> Dict[str, Any]:
        """
        Convert framework-specific output to universal format.

        Override this if your framework has special output format.

        Args:
            output: Framework-specific output

        Returns:
            Universal output dict
        """
        if isinstance(output, dict):
            return output
        return {"result": output}


class SyncToAsyncAdapter:
    """Helper to wrap sync agents for async execution."""

    def __init__(self, sync_callable: Callable):
        self.sync_callable = sync_callable

    async def __call__(self, *args, **kwargs):
        """Run sync callable in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.sync_callable(*args, **kwargs)
        )
