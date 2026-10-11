"""
Universal Executable for Any Framework

Wraps any framework's agent as a Sentinel Executable,
enabling 3-thread supervision (Actor|Observer|Governor).
"""

import asyncio
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from sentinal.core.executable import Executable
from sentinal.core.execution_context import ExecutionContext
from sentinal.core.execution_result import ExecutionResult
from .base import BaseFrameworkAdapter


class UniversalExecutable(Executable):
    """
    Universal executable that works with ANY framework.

    Wraps framework-specific agents using adapters,
    making them compatible with Sentinel's supervision system.

    Supports:
    - MAF
    - LangGraph
    - LangChain
    - CrewAI
    - Any custom framework with an adapter
    """

    def __init__(
        self,
        adapter: BaseFrameworkAdapter,
        quality_threshold: float = 0.75,
    ):
        """
        Initialize universal executable.

        Args:
            adapter: Framework-specific adapter
            quality_threshold: Minimum quality score (0-1)
        """
        self.adapter = adapter
        self.quality_threshold = quality_threshold

    async def execute(
        self,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute the agent with Sentinel supervision.

        This method is called by the Actor thread.
        The Observer and Governor threads monitor execution
        through the execution state.

        Args:
            context: Execution context from Sentinel

        Returns:
            Execution result with output and metadata
        """
        start_time = datetime.now(UTC)

        try:
            # Get input from context
            input_data = context.input

            # Execute through adapter
            output = await self.adapter.execute(input_data)

            # Extract quality metrics
            quality_metrics = self.adapter.extract_quality_metrics(output)

            # Calculate execution time
            end_time = datetime.now(UTC)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            # Build metadata
            metadata = {
                "agent_name": self.adapter.agent_name,
                "execution_time_ms": execution_time_ms,
                "quality_metrics": quality_metrics,
                "framework": self.adapter.__class__.__name__.replace("Adapter", ""),
            }

            # Return result
            return ExecutionResult(
                output=output,
                metadata=metadata,
            )

        except Exception as e:
            # Handle errors
            end_time = datetime.now(UTC)
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            return ExecutionResult(
                output=None,
                metadata={
                    "agent_name": self.adapter.agent_name,
                    "execution_time_ms": execution_time_ms,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
