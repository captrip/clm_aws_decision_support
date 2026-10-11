"""
Runtime Bridge - Integrates Sentinel Runtime with MAF Global_Orchestrator.

Manages the lifecycle of both systems and provides a unified entry point
for executing MAF workflows with Sentinel's reliability and observability.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
from uuid import UUID, uuid4
import asyncio

from sentinal.runtime.runtime import SentinelRuntime
from sentinal.core.actor_ref import ActorRef
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.snapshot import ExecutionStatus
from sentinal.policies.retry import RetryPolicy
from sentinal.policies.exponential import ExponentialRetryPolicy

from MAF.integration.executables import MAFWorkflowExecutable, SkillExecutable
from MAF.integration.skill_executor import SkillExecutor
from MAF.integration.state_bridge import StateManagerBridge


class MAFSentinelRuntimeBridge:
    """
    Unified runtime managing both Sentinel and MAF.
    
    Provides a single entry point for:
    - Starting/stopping Sentinel runtime
    - Registering skill executors as actors
    - Submitting MAF workflows for execution
    - Accessing execution results and observability data
    
    Usage:
        bridge = MAFSentinelRuntimeBridge(
            orchestrator=maf_orchestrator,
            retry_policy=ExponentialRetryPolicy(max_attempts=3)
        )
        
        # Start runtime
        await bridge.start()
        
        # Execute workflow
        result = await bridge.execute_workflow(
            user_query="...",
            memory_info={...}
        )
        
        # Stop runtime
        await bridge.stop()
    """
    
    def __init__(
        self,
        orchestrator: Any,  # Global_Orchestrator
        state_manager: Optional[Any] = None,
        retry_policy: Optional[RetryPolicy] = None,
        max_concurrent_skills: int = 4,
        auto_load_skills: bool = True,
        db_path: str = "subgraph_registry.db",
        SubgraphFactory: Optional[Any] = None,
        tools_registry: Optional[Any] = None,
    ):
        """
        Args:
            orchestrator: MAF Global_Orchestrator instance
            state_manager: MAF's StateManager
            retry_policy: Sentinel RetryPolicy for skill failures
                (defaults to ExponentialRetryPolicy)
            max_concurrent_skills: Max parallel skill executors
            auto_load_skills: Whether to lazy-load skills from DB
            db_path: Path to skill registry database
            SubgraphFactory: Factory for reconstructing skills
            tools_registry: Registry of available tools
        """
        self.orchestrator = orchestrator
        self.state_manager = state_manager or orchestrator.state_manager
        self.retry_policy = retry_policy or ExponentialRetryPolicy(
            max_attempts=3
        )
        self.max_concurrent_skills = max_concurrent_skills
        self.auto_load_skills = auto_load_skills
        self.db_path = db_path
        self.SubgraphFactory = SubgraphFactory
        self.tools_registry = tools_registry
        
        # Create Sentinel runtime
        self.sentinel_runtime = SentinelRuntime()
        
        # Track skill executors
        self.skill_executors: List[SkillExecutor] = []
        
        # Track execution sessions
        self.execution_sessions: Dict[UUID, ExecutionSession] = {}
    
    async def start(self) -> None:
        """Start the Sentinel runtime and register skill executors."""
        # Register SkillExecutor actors for concurrent skill execution
        for i in range(self.max_concurrent_skills):
            executor = SkillExecutor(ActorRef(f"maf_skill_executor_{i}", "local"))
            self.sentinel_runtime.register(executor)
            self.skill_executors.append(executor)

        await self.sentinel_runtime.start()
    
    async def stop(self) -> None:
        """Stop the Sentinel runtime."""
        await self.sentinel_runtime.stop()
    
    async def execute_workflow(
        self,
        user_query: str,
        memory_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        execution_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        Execute a complete MAF workflow with Sentinel orchestration.
        
        Args:
            user_query: User's input query
            memory_info: Optional memory/context dict
            metadata: Optional metadata to attach to execution
            execution_id: Optional custom execution ID (generated if not provided)
        
        Returns:
            Dictionary with:
                - response: Final user-facing response
                - execution_id: UUID of execution
                - execution_time_ms: Total time taken
                - status: 'completed' or 'failed'
                - state: Final MAF state
        """
        if not execution_id:
            execution_id = uuid4()
        
        # Create MAFWorkflowExecutable
        executable = MAFWorkflowExecutable(
            orchestrator=self.orchestrator,
            auto_load_skills=self.auto_load_skills,
            db_path=self.db_path,
            SubgraphFactory=self.SubgraphFactory,
            tools_registry=self.tools_registry,
            retry_policy=self.retry_policy,
        )
        
        # Create ExecutionContext
        context = ExecutionContext(
            input={
                "user_query": user_query,
                "memory_info": memory_info or {},
            },
            metadata=metadata or {},
        )
        
        # Create ExecutionSession
        session = ExecutionSession(
            executable=executable,
            execution_id=execution_id,
            context=context,
        )
        
        # Store session
        self.execution_sessions[execution_id] = session
        
        # Create bridge for state sync
        bridge = StateManagerBridge(
            state_manager=self.state_manager,
            execution_session=session,
        )
        
        # Sync MAF state to session
        bridge.sync_to_session()
        
        try:
            # Execute through Sentinel so its actors and observers manage the
            # workflow lifecycle.
            completed_session = await self.sentinel_runtime.execute(session)
            
            # Sync back from session
            bridge.sync_from_session()
            
            if completed_session.state.status is not ExecutionStatus.COMPLETED:
                return {
                    "response": None,
                    "error": completed_session.get_memory(
                        "error", "Sentinel execution failed"
                    ),
                    "execution_id": str(completed_session.execution_id),
                    "status": "failed",
                }

            result = completed_session.state.current.result
            if not isinstance(result, dict):
                raise TypeError(
                    "Sentinel completed the MAF workflow without a result mapping"
                )
            return result
        
        finally:
            # Clean up session tracking
            self.execution_sessions.pop(execution_id, None)
    
    async def execute_skill(
        self,
        skill_name: str,
        skill_entry: Dict[str, Any],
        state: Dict[str, Any],
        execution_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single skill via Sentinel actor.
        
        Args:
            skill_name: Name of the skill
            skill_entry: Dict with runner/graph/metadata
            state: Input state for skill
            execution_id: Optional custom execution ID
        
        Returns:
            Skill execution result from SkillExecutable
        """
        if not execution_id:
            execution_id = uuid4()
        
        # Create SkillExecutable
        executable = SkillExecutable(
            skill_name=skill_name,
            runner=skill_entry.get("runner"),
            graph=skill_entry.get("graph"),
            state_manager=self.state_manager,
            scope=skill_entry.get("scope", skill_name),
        )
        
        # Create ExecutionContext
        context = ExecutionContext(
            input=state,
            metadata={
                "skill_name": skill_name,
                "execution_id": str(execution_id),
            },
        )
        
        # Execute directly (could also submit as signal to actor)
        return await executable.execute(context)
    
    async def execute_skills_concurrent(
        self,
        skills: List[tuple[str, Dict[str, Any]]],
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple skills concurrently via SkillExecutor actors.
        
        Args:
            skills: List of (skill_name, skill_entry) tuples
            state: Shared state for all skills
        
        Returns:
            List of skill execution results
        """
        tasks = []
        
        for skill_name, skill_entry in skills:
            execution_id = uuid4()
            task = self.execute_skill(
                skill_name=skill_name,
                skill_entry=skill_entry,
                state=state,
                execution_id=execution_id,
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to result dicts
        formatted_results = []
        for result in results:
            if isinstance(result, Exception):
                formatted_results.append({
                    "skill_name": "unknown",
                    "error": str(result),
                    "executed": False,
                    "status": "failed",
                })
            else:
                formatted_results.append(result)
        
        return formatted_results
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about registered executors and sessions."""
        return {
            "registered_skill_executors": len(self.skill_executors),
            "active_sessions": len(self.execution_sessions),
            "retry_policy": type(self.retry_policy).__name__,
            "max_concurrent_skills": self.max_concurrent_skills,
        }
