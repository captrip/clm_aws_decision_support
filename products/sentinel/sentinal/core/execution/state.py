"""
ExecutionState - Manages execution lifecycle and state transitions

Tracks execution progress through events and maintains state history.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Dict, List

from sentinal.core.contracts.execution_events import (
    ExecutionStarted,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionPaused,
    ExecutionResumed,
    ExecutionCancelled,
)
from sentinal.core.trajectory import TrajectoryStep, Correction

from .snapshot import (
    ExecutionSnapshot,
    ExecutionStatus,
)
from .runtime_data import RuntimeData


class ExecutionState:
    """
    Mutable state machine for execution lifecycle.

    Tracks:
    - Current status (CREATED → RUNNING → COMPLETED/FAILED)
    - State history (snapshots)
    - Runtime memory (for Actor updates, Observer watching)
    - Instructions (from Governor to Actor)
    """

    def __init__(self):
        """Initialize execution state."""
        self._current = ExecutionSnapshot()
        self._history: List[ExecutionSnapshot] = [self._current]

        # Runtime memory - Actor writes, Observer reads
        self._runtime_memory: Dict[str, Any] = {}

        # Instructions - Governor writes, Actor reads
        self._instructions: List[str] = []

        # Trajectory - executable writes, Governor reads
        self._trajectory: List[TrajectoryStep] = []
        # Corrections - Governor writes, executable drains
        self._corrections: List[Correction] = []

        # Thread safety
        self._lock = Lock()

    @property
    def current(self) -> ExecutionSnapshot:
        """Get current snapshot."""
        with self._lock:
            return self._current

    @property
    def status(self) -> ExecutionStatus:
        """Get current status."""
        return self.current.status

    @property
    def history(self) -> List[ExecutionSnapshot]:
        """Get state history."""
        with self._lock:
            return self._history.copy()

    def apply(self, event) -> ExecutionSnapshot:
        """
        Apply event to state machine.

        Args:
            event: Execution event

        Returns:
            New snapshot after applying event
        """
        with self._lock:
            snapshot = self._current

            if isinstance(event, ExecutionStarted):
                snapshot = replace(
                    snapshot,
                    version=snapshot.version + 1,
                    status=ExecutionStatus.RUNNING,
                    updated_at=datetime.now(UTC),
                )

            elif isinstance(event, ExecutionCompleted):
                snapshot = replace(
                    snapshot,
                    version=snapshot.version + 1,
                    status=ExecutionStatus.COMPLETED,
                    result=event.data,
                    updated_at=datetime.now(UTC),
                )

            elif isinstance(event, ExecutionFailed):
                snapshot = replace(
                    snapshot,
                    version=snapshot.version + 1,
                    status=ExecutionStatus.FAILED,
                    updated_at=datetime.now(UTC),
                )

            elif isinstance(event, ExecutionPaused):
                snapshot = replace(
                    snapshot,
                    version=snapshot.version + 1,
                    status=ExecutionStatus.PAUSED,
                    updated_at=datetime.now(UTC),
                )

            elif isinstance(event, ExecutionResumed):
                snapshot = replace(
                    snapshot,
                    version=snapshot.version + 1,
                    status=ExecutionStatus.RUNNING,
                    updated_at=datetime.now(UTC),
                )

            elif isinstance(event, ExecutionCancelled):
                snapshot = replace(
                    snapshot,
                    version=snapshot.version + 1,
                    status=ExecutionStatus.CANCELLED,
                    updated_at=datetime.now(UTC),
                )

            self._history.append(snapshot)
            self._current = snapshot

            return snapshot

    # ==========================================
    # Runtime Memory (Actor updates, Observer watches)
    # ==========================================

    def update_memory(self, key: str, value: Any) -> None:
        """
        Update runtime memory.

        Called by Actor thread to report progress.
        Watched by Observer thread in real-time.

        Args:
            key: Memory key
            value: Memory value
        """
        with self._lock:
            self._runtime_memory[key] = value

    def get_memory(self, key: str, default=None) -> Any:
        """
        Get value from runtime memory.

        Args:
            key: Memory key
            default: Default value if key not found

        Returns:
            Memory value or default
        """
        with self._lock:
            return self._runtime_memory.get(key, default)

    def get_all_memory(self) -> Dict[str, Any]:
        """Get all runtime memory (snapshot)."""
        with self._lock:
            return self._runtime_memory.copy()

    # ==========================================
    # Instructions (Governor writes, Actor reads)
    # ==========================================

    def add_instruction(self, instruction: str) -> None:
        """
        Add instruction from Governor.

        Args:
            instruction: Instruction to add (RETRY, IMPROVE_QUALITY, etc.)
        """
        with self._lock:
            self._instructions.append(instruction)

    def get_instructions(self) -> List[str]:
        """
        Get all instructions (snapshot).

        Actor reads these to know what Governor wants.

        Returns:
            List of instructions
        """
        with self._lock:
            return self._instructions.copy()

    def clear_instructions(self) -> None:
        """Clear all instructions."""
        with self._lock:
            self._instructions.clear()

    # ==========================================
    # Trajectory (executable writes, Governor reads)
    # ==========================================

    def append_step(self, step: TrajectoryStep) -> None:
        with self._lock:
            self._trajectory.append(step)

    def get_trajectory(self) -> List[TrajectoryStep]:
        with self._lock:
            return self._trajectory.copy()

    # ==========================================
    # Corrections (Governor writes, executable drains)
    # ==========================================

    def push_correction(self, correction: Correction) -> None:
        with self._lock:
            self._corrections.append(correction)

    def drain_corrections(self) -> List[Correction]:
        with self._lock:
            drained = self._corrections.copy()
            self._corrections.clear()
            return drained

    # ==========================================
    # Metrics
    # ==========================================

    def add_metric(self, key: str, value: Any) -> None:
        """
        Add metric to current runtime data.

        Args:
            key: Metric key
            value: Metric value
        """
        with self._lock:
            new_runtime = self._current.runtime.with_metric(key, value)
            self._current = replace(
                self._current,
                runtime=new_runtime,
                updated_at=datetime.now(UTC),
            )

    def increment_retry(self) -> None:
        """Increment retry counter."""
        with self._lock:
            new_runtime = RuntimeData(
                current_node=self._current.runtime.current_node,
                retry_count=self._current.runtime.retry_count + 1,
                metrics=self._current.runtime.metrics,
                extras=self._current.runtime.extras,
            )
            self._current = replace(
                self._current,
                attempt=self._current.attempt + 1,
                runtime=new_runtime,
                updated_at=datetime.now(UTC),
            )

    # ==========================================
    # Status Checks
    # ==========================================

    def is_running(self) -> bool:
        """Check if execution is running."""
        return self.status == ExecutionStatus.RUNNING

    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self.status == ExecutionStatus.PAUSED

    def is_completed(self) -> bool:
        """Check if execution completed."""
        return self.status == ExecutionStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status == ExecutionStatus.FAILED

    def is_cancelled(self) -> bool:
        """Check if execution was cancelled."""
        return self.status == ExecutionStatus.CANCELLED

    def is_terminal(self) -> bool:
        """Check if execution is in terminal state."""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.ABORTED,
        ]
