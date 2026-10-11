"""
State Bridge - Bridges MAF's StateManager with Sentinel's ExecutionSession.

Provides bidirectional synchronization between:
- MAF's StateManager (scoped state for agents/skills)
- Sentinel's ExecutionSession (mutable runtime state)

This allows both systems to share state context while maintaining their
independent abstractions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from sentinal.core.execution.session import ExecutionSession


class StateManagerBridge:
    """
    Bridges MAF's StateManager with Sentinel's ExecutionSession.
    
    Usage:
        # Initialize bridge with both managers
        bridge = StateManagerBridge(
            state_manager=maf_state_mgr,
            execution_session=sentinel_session
        )
        
        # Sync MAF state → Sentinel ExecutionSession
        bridge.sync_to_session()
        
        # Sync Sentinel ExecutionSession → MAF state
        bridge.sync_from_session()
    """
    
    def __init__(
        self,
        state_manager: Any,  # MAF's StateManager
        execution_session: ExecutionSession,
    ):
        """
        Args:
            state_manager: MAF's StateManager instance
            execution_session: Sentinel's ExecutionSession instance
        """
        self.state_manager = state_manager
        self.execution_session = execution_session
    
    def sync_to_session(self) -> None:
        """
        Sync MAF's StateManager → Sentinel's ExecutionSession.
        
        Merges all MAF state (global + scopes) into Sentinel session memory.
        """
        if not self.execution_session or not self.state_manager:
            return
        
        # Get complete MAF state
        maf_state = self._get_complete_maf_state()
        
        self.execution_session.update_memory("maf_state", maf_state)
    
    def sync_from_session(self) -> None:
        """
        Sync Sentinel's ExecutionSession → MAF's StateManager.
        
        Extracts execution metadata from Sentinel session and stores in
        StateManager under 'sentinel' scope for agent access.
        """
        if not self.execution_session or not self.state_manager:
            return
        
        # Extract Sentinel execution metadata
        sentinel_data = {
            "execution_id": str(self.execution_session.execution_id),
            "status": self.execution_session.state.status.value,
            "updated_at": self.execution_session.state.current.updated_at.isoformat(),
        }
        
        # Store in MAF state
        self.state_manager.update("sentinel", sentinel_data)
    
    def store_skill_result(
        self,
        skill_name: str,
        result: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        """
        Store a skill's execution result in both systems.
        
        Args:
            skill_name: Name of the skill
            result: Result dict from SkillExecutable
            error: Optional error string if execution failed
        """
        skill_data = {
            "result": result.get("result"),
            "execution_time_ms": result.get("execution_time_ms"),
            "status": result.get("status"),
            "executed": result.get("executed", True),
        }
        
        if error:
            skill_data["error"] = error
        
        # Store in MAF StateManager
        self.state_manager.update(skill_name, skill_data)
        
        skill_results = self.execution_session.get_memory("skill_results", {})
        skill_results[skill_name] = skill_data
        self.execution_session.update_memory("skill_results", skill_results)
    
    def get_combined_state(self) -> Dict[str, Any]:
        """
        Get combined view of MAF state + Sentinel metadata.
        
        Returns:
            Dictionary with keys:
                - maf: Complete MAF state
                - sentinel: Sentinel execution metadata
        """
        return {
            "maf": self._get_complete_maf_state(),
            "sentinel": {
                "execution_id": str(self.execution_session.execution_id),
                "session_id": str(self.execution_session.session_id),
                "status": self.execution_session.state.status.value,
                "runtime_memory": self.execution_session.state.get_all_memory(),
            },
        }
    
    def _get_complete_maf_state(self) -> Dict[str, Any]:
        """Get complete MAF state (global + all scopes)."""
        if not self.state_manager:
            return {}
        
        complete_state = {}
        
        if hasattr(self.state_manager, "get_global"):
            complete_state["global"] = self.state_manager.get_global()

        if hasattr(self.state_manager, "list_scopes"):
            complete_state["scopes"] = {
                scope_name: self.state_manager.get_scope(scope_name)
                for scope_name in self.state_manager.list_scopes()
            }
        
        return complete_state


class ExecutionSessionWrapper:
    """
    Wraps a Sentinel ExecutionSession to provide MAF-style state access.
    
    Allows MAF agents to read/write execution state as if it were a dict,
    while being backed by Sentinel's ExecutionSession.
    """
    
    def __init__(self, session: ExecutionSession):
        self.session = session
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session runtime memory."""
        if key in self._cache:
            return self._cache[key]
        
        return self.session.get_memory(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in session memory and cache."""
        self._cache[key] = value
        self.session.update_memory(key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all session data as dict."""
        base = self.session.state.get_all_memory()
        base.update(self._cache)
        return base
