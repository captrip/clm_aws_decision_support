"""
MAF + Sentinel Integration Module

Provides unified orchestration, observability, and reliability features
by integrating MAF's LLM-driven workflows with Sentinel's actor framework.

Key Components:
- MAFWorkflowExecutable: Wraps Global_Orchestrator as Sentinel Executable
- SkillExecutable: Wraps individual skills for Sentinel execution
- SkillExecutor: Sentinel actor for concurrent skill execution
- StateManagerBridge: Syncs state between MAF and Sentinel
- MAFSentinelRuntimeBridge: Unified runtime entry point

Usage Example:
    from MAF.integration import MAFSentinelRuntimeBridge
    from MAF.global_.orchestrator_graph import Global_Orchestrator
    
    orchestrator = Global_Orchestrator()
    bridge = MAFSentinelRuntimeBridge(orchestrator)
    
    await bridge.start()
    result = await bridge.execute_workflow(
        user_query="What can you help with?",
        memory_info={}
    )
    await bridge.stop()
"""

from MAF.integration.executables import (
    MAFWorkflowExecutable,
    SkillExecutable,
)
from MAF.integration.skill_executor import SkillExecutor, ExecuteSkillSignal
from MAF.integration.state_bridge import (
    StateManagerBridge,
    ExecutionSessionWrapper,
)
from MAF.integration.runtime_bridge import MAFSentinelRuntimeBridge

__all__ = [
    # Executables
    "MAFWorkflowExecutable",
    "SkillExecutable",
    # Actors
    "SkillExecutor",
    "ExecuteSkillSignal",
    # State Management
    "StateManagerBridge",
    "ExecutionSessionWrapper",
    # Runtime
    "MAFSentinelRuntimeBridge",
]
