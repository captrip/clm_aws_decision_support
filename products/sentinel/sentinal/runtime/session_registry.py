from __future__ import annotations

from uuid import UUID

from sentinal.core.execution.session import ExecutionSession


class SessionRegistry:
    """
    In-memory registry for active execution sessions.

    The runtime is the sole owner of this registry.
    """

    def __init__(self) -> None:
        self._sessions: dict[UUID, ExecutionSession] = {}

    def add(
        self,
        session: ExecutionSession,
    ) -> None:
        self._sessions[session.execution_id] = session

    def get(
        self,
        execution_id: UUID,
    ) -> ExecutionSession:
        return self._sessions[execution_id]

    def remove(
        self,
        execution_id: UUID,
    ) -> None:
        self._sessions.pop(execution_id, None)

    def exists(
        self,
        execution_id: UUID,
    ) -> bool:
        return execution_id in self._sessions

    def clear(self) -> None:
        """Clear all sessions."""
        self._sessions.clear()

    def list_all(self) -> list[ExecutionSession]:
        """Get all sessions."""
        return list(self._sessions.values())