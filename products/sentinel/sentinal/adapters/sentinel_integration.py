"""
Sentinel Integration with Universal Adapters

Connects the complete Sentinel runtime (Actor|Observer|Governor)
with universal framework adapters (MAF, LangGraph, LangChain, CrewAI).

This is the bridge between:
- Sentinel's 3-thread supervision
- Framework-agnostic adapters
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from sentinal.runtime.runtime import SentinelRuntime
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.execution.context import ExecutionContext
from sentinal.adapters.universal_executable import UniversalExecutable
from sentinal.adapters.base import BaseFrameworkAdapter
from sentinal.adapters.maf_adapter import MAFAdapter
from sentinal.adapters.langgraph_adapter import LangGraphAdapter
from sentinal.adapters.langchain_adapter import LangChainAdapter
from sentinal.adapters.crewai_adapter import CrewAIAdapter

FrameworkType = Literal["maf", "langgraph", "langchain", "crewai", "auto"]


class SupervisedAgent:
    """
    Universal supervised agent powered by complete Sentinel.

    Wraps any framework's agent with:
    - Actor thread (executes)
    - Observer thread (watches memory)
    - Governor thread (monitors quality, intervenes)

    Usage:
        from sentinal.adapters.sentinel_integration import supervise

        supervised = supervise(your_agent, framework="maf")
        result = await supervised.execute({"query": "test"})
    """

    def __init__(
        self,
        agent: Any,
        adapter: BaseFrameworkAdapter,
        quality_threshold: float = 0.75,
        enable_supervision: bool = True,
    ):
        """
        Initialize supervised agent.

        Args:
            agent: Framework-specific agent
            adapter: Framework adapter
            quality_threshold: Quality threshold for Governor
            enable_supervision: Enable/disable 3-thread supervision
        """
        self.agent = agent
        self.adapter = adapter
        self.quality_threshold = quality_threshold
        self.enable_supervision = enable_supervision

        # Create Sentinel runtime
        self.runtime = SentinelRuntime(quality_threshold=quality_threshold)

        # Stats
        self.execution_count = 0

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent with Sentinel 3-thread supervision.

        Args:
            input_data: Universal input dict

        Returns:
            Result with supervision metadata
        """
        self.execution_count += 1

        if not self.enable_supervision:
            # No supervision - direct execution
            result = await self.adapter.execute(input_data)
            return {
                "result": result,
                "supervision_enabled": False,
                "execution_count": self.execution_count,
            }

        # Create executable from adapter
        executable = UniversalExecutable(
            adapter=self.adapter,
            quality_threshold=self.quality_threshold,
        )

        # Create execution context
        context = ExecutionContext(input=input_data)

        # Create session
        session = ExecutionSession(
            executable=executable,
            context=context,
        )

        # Execute with Sentinel (3 threads: Actor|Observer|Governor)
        completed_session = await self.runtime.execute(session)

        # Extract results
        memory = completed_session.state.get_all_memory()
        result_data = memory.get("result")

        # Get actual output
        output = None
        if result_data and hasattr(result_data, 'output'):
            output = result_data.output

        # Build response
        return {
            "result": output,
            "metadata": {
                "execution_id": str(completed_session.execution_id),
                "status": str(completed_session.state.status),
                "agent_name": self.adapter.agent_name,
                "framework": self.adapter.__class__.__name__.replace("Adapter", ""),
            },
            "supervision": {
                "observer_observations": memory.get("observations", []),
                "governor_interventions": completed_session.get_instructions(),
                "runtime_memory": memory,
            },
            "supervision_enabled": True,
            "execution_count": self.execution_count,
        }

    async def __call__(self, input_data: Dict[str, Any]) -> Any:
        """
        Make agent callable.

        Returns just the result (not metadata) for drop-in compatibility.

        Args:
            input_data: Input data

        Returns:
            Agent result
        """
        full_result = await self.execute(input_data)
        return full_result.get("result")

    async def shutdown(self) -> None:
        """Shutdown runtime."""
        await self.runtime.shutdown()


def detect_framework(agent: Any) -> str:
    """
    Auto-detect framework from agent.

    Args:
        agent: Agent instance

    Returns:
        Framework name
    """
    agent_class = agent.__class__.__name__
    agent_module = agent.__class__.__module__

    # Check module path
    if "maf" in agent_module.lower() or "MAFI" in agent_module:
        return "maf"

    if "langgraph" in agent_module.lower():
        return "langgraph"

    if "langchain" in agent_module.lower():
        return "langchain"

    if "crewai" in agent_module.lower() or "crew" in agent_module.lower():
        return "crewai"

    # Check class name
    if "MAF" in agent_class or "Intent" in agent_class:
        return "maf"

    if "Graph" in agent_class or "State" in agent_class:
        return "langgraph"

    if "Chain" in agent_class or "Agent" in agent_class:
        return "langchain"

    if "Crew" in agent_class:
        return "crewai"

    return "unknown"


def supervise(
    agent: Any,
    framework: FrameworkType = "auto",
    agent_name: str = None,
    quality_threshold: float = 0.75,
    enable_supervision: bool = True,
) -> SupervisedAgent:
    """
    Wrap any agent with complete Sentinel 3-thread supervision.

    This is the main entry point for universal supervision.

    Args:
        agent: Agent from any framework
        framework: Framework type or "auto"
        agent_name: Human-readable name
        quality_threshold: Quality threshold (0-1)
        enable_supervision: Enable/disable supervision

    Returns:
        SupervisedAgent with 3-thread execution

    Examples:
        >>> # MAF
        >>> from MAFI.agents.IU_agents import IntentUnderstandingAgent
        >>> supervised = supervise(IntentUnderstandingAgent(), framework="maf")
        >>> result = await supervised.execute({"query": "test"})

        >>> # LangGraph
        >>> supervised = supervise(graph, framework="langgraph")

        >>> # Auto-detect
        >>> supervised = supervise(any_agent)
    """

    # Auto-detect framework
    if framework == "auto":
        framework = detect_framework(agent)
        if framework == "unknown":
            raise ValueError(
                f"Could not auto-detect framework for {agent.__class__.__name__}. "
                "Specify framework explicitly: supervise(agent, framework='maf')"
            )

    # Get adapter class
    adapter_map = {
        "maf": MAFAdapter,
        "langgraph": LangGraphAdapter,
        "langchain": LangChainAdapter,
        "crewai": CrewAIAdapter,
    }

    if framework not in adapter_map:
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Supported: {list(adapter_map.keys())}"
        )

    # Create adapter
    if agent_name is None:
        agent_name = agent.__class__.__name__

    adapter = adapter_map[framework](agent, agent_name)

    # Create supervised agent
    return SupervisedAgent(
        agent=agent,
        adapter=adapter,
        quality_threshold=quality_threshold,
        enable_supervision=enable_supervision,
    )
