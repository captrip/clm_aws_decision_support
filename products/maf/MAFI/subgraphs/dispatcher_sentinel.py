"""
Sentinel-integrated skill dispatcher for MAF.

Replaces the concurrent asyncio approach with Sentinel-backed execution,
providing Governor-driven retries, better observability, and state management.

Can be used alongside the original dispatcher_concurrent.py for gradual migration.
"""

from typing import Dict, Any, List, Optional
from uuid import uuid4

from MAFI.agent_factory.types.state_manager import StateManager
from MAFI.subgraphs.subgraph_registry import SubgraphRegistry
from MAFI.subgraphs.registry_sync import RegistrySync

from MAF.integration import MAFSentinelRuntimeBridge


async def dispatch_selected_skill_sentinel(
    state: Dict[str, Any],
    runtime_bridge: MAFSentinelRuntimeBridge,
    auto_load_from_db: bool = True,
    db_path: str = "subgraph_registry.db",
    SubgraphFactory: Optional[Any] = None,
    tools_registry: Optional[Any] = None,
    state_manager: Optional[StateManager] = None,
) -> Dict[str, Any]:
    """
    Execute selected skills using Sentinel-backed execution.
    
    Replaces dispatch_selected_skill for Sentinel-integrated MAF workflows.
    Provides:
    - Governor-driven retry policies
    - Better observability via ExecutionSession
    - Scoped state management via StateManagerBridge
    - Built-in error tracking and metrics
    
    Args:
        state: Current MAF state dict
        runtime_bridge: MAFSentinelRuntimeBridge instance (must be started)
        auto_load_from_db: Whether to lazy-load skills from DB
        db_path: Path to skill registry database
        SubgraphFactory: Factory for reconstructing skills
        tools_registry: Registry of available tools
        state_manager: MAF's StateManager instance
    
    Returns:
        Dictionary with keys:
            - executions: List of execution results
            - executed_skills: List of successfully executed skill names
            - tool_results_summary: List of tool output summaries
    """
    selected = state.get("selected_skills", [])
    if not selected:
        return {"executions": [], "executed_skills": [], "tool_results_summary": []}
    
    # Get current skill registry snapshot
    registry_snapshot = SubgraphRegistry.list_all()
    
    # Load missing skills from DB if needed
    if auto_load_from_db:
        already_registered = [s for s in selected if s in registry_snapshot]
        missing_skills = [s for s in selected if s not in registry_snapshot]
        
        if missing_skills:
            if SubgraphFactory is not None and tools_registry is not None:
                sync = RegistrySync(db_path=db_path)
                try:
                    for skill_name in missing_skills:
                        success = sync.recreate_subgraph_from_db(
                            name=skill_name,
                            SubgraphFactory=SubgraphFactory,
                            tools_registry=tools_registry,
                            state_mgr=state_manager,
                        )
                        if success:
                            print(f"[Sentinel] Loaded skill: {skill_name}")
                        else:
                            print(f"[Sentinel] Skill not found in DB: {skill_name}")
                finally:
                    sync.close()
                registry_snapshot = SubgraphRegistry.list_all()
    
    # Prepare skills for concurrent execution
    skills_to_execute: List[tuple[str, Dict[str, Any]]] = []
    
    for skill_name in selected:
        entry = registry_snapshot.get(skill_name)
        if not entry:
            print(f"[Sentinel] Skill not registered: {skill_name}")
            continue
        skills_to_execute.append((skill_name, entry))
    
    # Execute all skills concurrently via Sentinel runtime
    executions = await runtime_bridge.execute_skills_concurrent(
        skills=skills_to_execute,
        state=state,
    )
    
    # Process results (match original dispatcher output format)
    executed_names: List[str] = []
    tool_summaries: List[str] = []
    
    for execution in executions:
        if execution.get("executed", False):
            executed_names.append(execution.get("skill_name", "unknown"))
        
        # Extract tool summaries from result
        result_payload = execution.get("result", {}) or {}
        sub_summaries = result_payload.get("tool_results_summary") or []
        
        if isinstance(sub_summaries, str):
            tool_summaries.append(sub_summaries)
        elif isinstance(sub_summaries, list):
            tool_summaries.extend(str(s) for s in sub_summaries)
    
    # Calculate metrics
    successful_count = len([e for e in executions if e.get("executed", False)])
    failed_count = len(executions) - successful_count
    total_duration = sum(e.get("execution_time_ms", 0) for e in executions)
    
    print(
        f"[Sentinel] Skill execution summary: "
        f"{successful_count} succeeded, {failed_count} failed, "
        f"{total_duration:.2f}ms total"
    )
    
    # Convert to original dispatcher format for compatibility
    formatted_executions = []
    for execution in executions:
        formatted = {
            "skill": execution.get("skill_name", "unknown"),
            "executed": execution.get("executed", False),
            "duration_ms": execution.get("execution_time_ms", 0),
            "timestamp": execution.get("timestamp", 0),
        }
        
        if execution.get("status") == "completed":
            formatted["result"] = execution.get("result")
        else:
            formatted["error"] = execution.get("error", "Unknown error")
        
        formatted_executions.append(formatted)
    
    return {
        "executions": formatted_executions,
        "executed_skills": executed_names,
        "tool_results_summary": tool_summaries,
        "sentinel_metrics": {
            "successful": successful_count,
            "failed": failed_count,
            "total_duration_ms": round(total_duration, 2),
            "retry_policy": type(runtime_bridge.retry_policy).__name__,
        },
    }
