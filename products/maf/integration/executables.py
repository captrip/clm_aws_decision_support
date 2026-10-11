"""
Executable wrappers for MAF workflows and skills to run within Sentinel framework.

Converts MAF's LangGraph workflows into Sentinel Executables for orchestrated,
observable, and recoverable execution with built-in retry policies.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, Callable, Awaitable

from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.runtime_data import RuntimeData
from sentinal.policies.retry import RetryPolicy


class MAFWorkflowExecutable(Executable):
    """
    Wraps the Global_Orchestrator as a Sentinel Executable.
    
    Receives a user query + memory context via ExecutionContext.input,
    invokes the complete MAF workflow, and returns the final response.
    
    Usage:
        executable = MAFWorkflowExecutable(orchestrator)
        context = ExecutionContext(
            input={
                "user_query": "...",
                "memory_info": {...}
            }
        )
        result = await executable.execute(context)
    """
    
    def __init__(
        self,
        orchestrator,  # Global_Orchestrator instance
        auto_load_skills: bool = True,
        db_path: str = "subgraph_registry.db",
        SubgraphFactory: Optional[Any] = None,
        tools_registry: Optional[Any] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        """
        Args:
            orchestrator: Global_Orchestrator instance
            auto_load_skills: Whether to lazy-load skills from DB
            db_path: Path to skill registry database
            SubgraphFactory: Factory for reconstructing skills from DB
            tools_registry: Registry of available tools
        """
        self.orchestrator = orchestrator
        self.auto_load_skills = auto_load_skills
        self.db_path = db_path
        self.SubgraphFactory = SubgraphFactory
        self.tools_registry = tools_registry
        self.retry_policy = retry_policy
    
    async def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute the MAF workflow with the provided context.
        
        Args:
            context: ExecutionContext containing user_query and memory_info
        
        Returns:
            Dictionary with keys:
                - response: Final user-facing response
                - execution_time_ms: Total execution time
                - state: Final orchestrator state
                - execution_id: UUID of this execution
        """
        execution_start_time = time.perf_counter()
        
        # Extract input from ExecutionContext
        input_data = context.input or {}
        user_query = input_data.get("user_query", "")
        memory_info = input_data.get("memory_info", {})
        
        # Build initial state for orchestrator
        session = context.session
        initial_state = {
            "user_query": user_query,
            "memory_info": memory_info,
            "session_id": str(session.session_id) if session else "",
            "execution_id": str(session.execution_id) if session else "",
            **context.metadata,
        }
        
        retry_count = 0
        while True:
            try:
                result = await self.orchestrator.graph.ainvoke(
                    initial_state,
                    config={"recursion_limit": 100},
                )
                execution_time_ms = (
                    time.perf_counter() - execution_start_time
                ) * 1000
                return {
                    "response": result.get("summary", ""),
                    "execution_time_ms": round(execution_time_ms, 2),
                    "state": result,
                    "execution_id": initial_state["execution_id"],
                    "status": "completed",
                }
            except Exception as error:
                if self.retry_policy is None:
                    raise

                runtime_data = RuntimeData(retry_count=retry_count)
                if not self.retry_policy.should_retry(runtime_data, error):
                    raise

                await asyncio.sleep(
                    self.retry_policy.next_delay(runtime_data)
                )
                retry_count += 1


class SkillExecutable(Executable):
    """
    Wraps an individual skill (runner or graph) as a Sentinel Executable.
    
    Used by SkillExecutor actor to execute skills with state management
    and error tracking.
    
    Usage:
        executable = SkillExecutable(
            skill_name="web_search",
            runner=search_runner,
            state_manager=state_mgr
        )
        context = ExecutionContext(input={...})
        result = await executable.execute(context)
    """
    
    def __init__(
        self,
        skill_name: str,
        runner: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
        graph: Optional[Any] = None,
        state_manager: Optional[Any] = None,
        scope: Optional[str] = None,
    ):
        """
        Args:
            skill_name: Name of the skill
            runner: Async callable that executes the skill
            graph: LangGraph subgraph for the skill
            state_manager: StateManager for storing scoped results
            scope: State scope key (defaults to skill_name)
        """
        self.skill_name = skill_name
        self.runner = runner
        self.graph = graph
        self.state_manager = state_manager
        self.scope = scope or skill_name
    
    async def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute the skill with the provided state.
        
        Args:
            context: ExecutionContext containing input state
        
        Returns:
            Dictionary with:
                - result: Skill execution output
                - skill_name: Name of the skill
                - execution_time_ms: Time taken
                - execution_id: UUID of this execution
        """
        start_time = time.time()
        
        try:
            # Prepare enriched state
            state = context.input or {}
            enriched_state = dict(state)
            
            # Add state manager context if available
            if self.state_manager:
                enriched_state["__context"] = (
                    self.state_manager.context().to_dict()
                )
            
            # Execute skill (runner or graph)
            if self.runner is not None:
                result = await self.runner(enriched_state)
            elif self.graph is not None:
                result = await self.graph.ainvoke(enriched_state)
            else:
                raise ValueError(
                    f"Skill '{self.skill_name}' has no runner or graph"
                )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Store result in state manager if available
            if self.state_manager:
                self.state_manager.update(
                    self.scope,
                    {
                        "execution_result": result,
                        "execution_time_ms": round(execution_time_ms, 2),
                    }
                )
            
            return {
                "result": result,
                "skill_name": self.skill_name,
                "execution_time_ms": round(execution_time_ms, 2),
                "execution_id": str(
                    context.session.execution_id
                    if context.session
                    else context.metadata.get("execution_id", "")
                ),
                "executed": True,
                "status": "completed",
            }
        
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Store error in state manager
            if self.state_manager:
                self.state_manager.update(
                    self.scope,
                    {
                        "execution_error": str(e),
                        "execution_time_ms": round(execution_time_ms, 2),
                    }
                )
            
            return {
                "result": None,
                "error": str(e),
                "skill_name": self.skill_name,
                "execution_time_ms": round(execution_time_ms, 2),
                "execution_id": str(
                    context.session.execution_id
                    if context.session
                    else context.metadata.get("execution_id", "")
                ),
                "executed": False,
                "status": "failed",
            }
