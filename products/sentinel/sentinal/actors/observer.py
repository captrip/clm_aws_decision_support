"""
Observer - Watches runtime memory in real-time

Responsibilities:
- Poll runtime_memory for changes
- Log state transitions
- Collect metrics
- Publish observations
- Detect anomalies
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict

from sentinal.actors.actor import BaseActor
from sentinal.telemetry.dashboard import activity_bus

from sentinal.core.contracts.execution_events import (
    ExecutionCompleted,
    ExecutionFailed,
)

from sentinal.core.contracts.execution_observations import (
    ExecutionSucceeded,
    ExecutionErrored,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentinal.core.execution.session import ExecutionSession


class Observer(BaseActor):
    """
    Observer actor - watches execution in real-time.

    Monitors:
    - runtime_memory changes (what Actor updates)
    - Execution progress
    - Quality metrics
    - Anomalies

    Publishes observations for Governor to consume.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observations = []
        self._last_memory_snapshot = {}

    async def receive(
        self,
        message,
    ):
        """
        Receive execution events.

        Args:
            message: Incoming event
        """
        if isinstance(
            message,
            ExecutionCompleted,
        ):
            await self.publish(
                ExecutionSucceeded(
                    execution_id=message.execution_id,
                    sender=self.ref,
                    data=None,
                )
            )
            return

        if isinstance(
            message,
            ExecutionFailed,
        ):
            await self.publish(
                ExecutionErrored(
                    execution_id=message.execution_id,
                    sender=self.ref,
                    data=message.data,
                )
            )

    async def observe_session(self, session: 'ExecutionSession', poll_interval: float = 0.05) -> None:
        """
        Observe a session in real-time.

        Polls runtime_memory for changes while execution is running.

        Args:
            session: Execution session to observe
            poll_interval: How often to check (seconds)
        """
        logger.info(f"Observer {self.ref.name} started observing")

        sid = str(session.session_id)
        activity_bus.emit("Observer", "observing", status="observing", session_id=sid)

        observation_count = 0
        last_status = None

        while not session.is_terminal():
            # Get current memory snapshot
            current_memory = session.state.get_all_memory()

            # Check for changes
            changes = self._detect_changes(self._last_memory_snapshot, current_memory)

            if changes:
                observation_count += 1

                # Log changes
                for key, (old_val, new_val) in changes.items():
                    self._observations.append({
                        "key": key,
                        "old_value": old_val,
                        "new_value": new_val,
                        "observation_number": observation_count,
                    })
                    activity_bus.emit("Observer", "change.detected",
                                      status="observing", detail=key, session_id=sid)

                # Check actor status changes
                status = current_memory.get("actor_status")
                if status != last_status:
                    logger.debug(f"Observer detected status change: {status}")
                    last_status = status

                self._last_memory_snapshot = current_memory.copy()

            # Poll interval
            await asyncio.sleep(poll_interval)

        activity_bus.emit("Observer", "complete", status="complete",
                          detail=str(observation_count), session_id=sid)
        logger.info(f"Observer {self.ref.name} complete ({observation_count} observations)")

    def _detect_changes(
        self,
        old_memory: Dict[str, Any],
        new_memory: Dict[str, Any],
    ) -> Dict[str, tuple]:
        """
        Detect changes between memory snapshots.

        Args:
            old_memory: Previous memory state
            new_memory: Current memory state

        Returns:
            Dictionary of changes: {key: (old_value, new_value)}
        """
        changes = {}

        # Check for new or modified keys
        for key, new_val in new_memory.items():
            old_val = old_memory.get(key)
            if old_val != new_val:
                changes[key] = (old_val, new_val)

        # Check for removed keys
        for key in old_memory:
            if key not in new_memory:
                changes[key] = (old_memory[key], None)

        return changes

    def get_observations(self) -> list:
        """Get all observations collected."""
        return self._observations.copy()

    def compute_quality_score(self, session: 'ExecutionSession') -> float:
        """
        Compute quality score based on observations.

        Args:
            session: Execution session

        Returns:
            Quality score (0-1)
        """
        memory = session.state.get_all_memory()

        # Check if execution completed successfully
        if memory.get("actor_status") != "completed":
            return 0.3

        # Check if result exists
        if "result" not in memory:
            return 0.4

        # Check for errors
        if "error" in memory:
            return 0.2

        # Base quality for successful completion
        return 0.85
