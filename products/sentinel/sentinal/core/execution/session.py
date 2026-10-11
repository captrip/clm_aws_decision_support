"""
ExecutionSession - Aggregate root for a single execution

Owns everything required to execute and track a workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from sentinal.core.executable import Executable

from .context import ExecutionContext
from .state import ExecutionState


@dataclass(slots=True)
class ExecutionSession:
    """
    Aggregate root for a single execution.

    Combines:
    - Executable (what to run)
    - Context (input data)
    - State (runtime state machine)

    This is what gets passed between actors.
    """

    executable: Executable[Any, Any]
    """The executable to run"""

    context: ExecutionContext
    """Input context"""

    state: ExecutionState = field(default_factory=ExecutionState)
    """Runtime state (mutable)"""

    execution_id: UUID = field(default_factory=uuid4)
    """Unique execution ID"""

    session_id: UUID = field(default_factory=uuid4)
    """Session ID (for grouping related executions)"""

    # ==========================================
    # Convenience Methods
    # ==========================================

    def update_memory(self, key: str, value: Any) -> None:
        """Update runtime memory (Actor → Observer)."""
        self.state.update_memory(key, value)

    def get_memory(self, key: str, default=None) -> Any:
        """Read runtime memory."""
        return self.state.get_memory(key, default)

    def add_instruction(self, instruction: str) -> None:
        """Add instruction (Governor → Actor)."""
        self.state.add_instruction(instruction)

    def get_instructions(self) -> list[str]:
        """Get all instructions."""
        return self.state.get_instructions()

    def append_step(self, step) -> None:
        """Append a trajectory step (executable → Governor)."""
        self.state.append_step(step)

    def get_trajectory(self) -> list:
        """Get trajectory snapshot (Governor reads)."""
        return self.state.get_trajectory()

    def push_correction(self, correction) -> None:
        """Push a correction (Governor → executable)."""
        self.state.push_correction(correction)

    def drain_corrections(self) -> list:
        """Drain pending corrections (executable reads & clears)."""
        return self.state.drain_corrections()

    def is_running(self) -> bool:
        """Check if execution is running."""
        return self.state.is_running()

    def is_terminal(self) -> bool:
        """Check if execution finished (completed/failed/cancelled)."""
        return self.state.is_terminal()
