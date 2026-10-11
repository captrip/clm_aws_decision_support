"""
Executor - Actor thread that executes workloads

Responsibilities:
- Execute the Executable
- Update runtime memory as it runs (for Observer)
- Check instructions from Governor
- Handle signals (pause/resume/cancel)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from dataclasses import replace

from sentinal.actors.actor import BaseActor
from sentinal.telemetry.dashboard import activity_bus
from sentinal.core.contracts.execution_events import (
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionStarted,
    ExecutionPaused,
    ExecutionResumed,
    ExecutionCancelled,
)
from sentinal.core.contracts.execution_signals import (
    StartExecution,
)
from sentinal.exceptions.runtime import ExecutionAborted

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentinal.core.execution.session import ExecutionSession


class Executor(BaseActor):
    """
    Executor actor - the Actor thread.

    Receives StartExecution signals and invokes an Executable.

    While executing:
    - Updates runtime_memory (Observer watches this)
    - Checks instructions from Governor
    - Can be paused/resumed/cancelled
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_task = None
        self._paused = False
        self._cancelled = False

    async def receive(self, message) -> None:
        """
        Receive and handle messages.

        Args:
            message: Incoming message
        """
        if isinstance(message, StartExecution):
            await self._handle_start_execution(message)

    async def _handle_start_execution(self, message: StartExecution) -> None:
        """
        Handle StartExecution signal.

        Args:
            message: StartExecution message
        """
        # Get session from message data (execution_id)
        # For now, we'll need the session passed differently
        # This will be fixed when we integrate with runtime

        # TODO: Get session from registry
        # For now, assume message.data contains session
        if not hasattr(message, 'data') or message.data is None:
            return

        execution_id = message.data

        # This will be properly wired when we complete the runtime
        # For now, this is the structure

    async def execute_session(self, session: 'ExecutionSession') -> None:
        """
        Execute a session with full supervision.

        Args:
            session: Execution session to run
        """
        logger.info(f"Executor starting session {session.execution_id}")

        sid = str(session.session_id)

        # Update memory: Actor starting
        session.update_memory("actor_status", "starting")
        session.update_memory("actor_thread_id", id(asyncio.current_task()))
        activity_bus.emit("Executor", "started", status="running", session_id=sid)

        # Apply state transition
        from sentinal.core.execution.snapshot import ExecutionStatus
        from datetime import datetime, UTC

        # Transition to RUNNING
        snapshot = session.state._current

        new_snapshot = replace(
            snapshot,
            status=ExecutionStatus.RUNNING,
            updated_at=datetime.now(UTC),
        )
        session.state._current = new_snapshot
        session.state._history.append(new_snapshot)

        # Publish ExecutionStarted event
        await self.publish(
            ExecutionStarted(
                execution_id=session.execution_id,
                sender=self.ref,
                data=None,
            )
        )

        # Update memory: Actor executing
        session.update_memory("actor_status", "executing")

        try:
            # Check for instructions before starting
            instructions = session.get_instructions()
            if instructions:
                session.update_memory("following_instructions", instructions)

            # Attach the live session so streaming executables can steer
            context = replace(session.context, session=session)
            # Execute the workload
            activity_bus.emit("Executor", "executing", status="running", session_id=sid)
            result = await session.executable.execute(context)

            # Update memory: Actor completed
            session.update_memory("actor_status", "completed")
            session.update_memory("result", result)

            # Transition to COMPLETED
            new_snapshot = replace(
                session.state._current,
                status=ExecutionStatus.COMPLETED,
                result=result,
                updated_at=datetime.now(UTC),
            )
            session.state._current = new_snapshot
            session.state._history.append(new_snapshot)

            # Publish ExecutionCompleted event
            await self.publish(
                ExecutionCompleted(
                    execution_id=session.execution_id,
                    sender=self.ref,
                    data=result,
                )
            )

            activity_bus.emit("Executor", "completed", status="completed", session_id=sid)
            logger.info(f"Executor completed session {session.execution_id}")

        except ExecutionAborted as ex:
            logger.warning(f"Executor aborted by governor: {ex}")
            session.update_memory("actor_status", "aborted")
            session.update_memory("abort_reason", str(ex))

            new_snapshot = replace(
                session.state._current,
                status=ExecutionStatus.ABORTED,
                updated_at=datetime.now(UTC),
            )
            session.state._current = new_snapshot
            session.state._history.append(new_snapshot)

            await self.publish(
                ExecutionFailed(
                    execution_id=session.execution_id,
                    sender=self.ref,
                    data=ex,
                )
            )

            activity_bus.emit("Executor", "aborted", status="aborted",
                              detail=str(ex), session_id=sid)

        except asyncio.CancelledError:
            # Execution was cancelled
            session.update_memory("actor_status", "cancelled")

            new_snapshot = replace(
                session.state._current,
                status=ExecutionStatus.CANCELLED,
                updated_at=datetime.now(UTC),
            )
            session.state._current = new_snapshot
            session.state._history.append(new_snapshot)

            await self.publish(
                ExecutionCancelled(
                    execution_id=session.execution_id,
                    sender=self.ref,
                    data=None,
                )
            )
            raise

        except Exception as ex:
            # Execution failed
            logger.error(f"Executor failed: {ex}")
            session.update_memory("actor_status", "failed")
            session.update_memory("error", str(ex))

            new_snapshot = replace(
                session.state._current,
                status=ExecutionStatus.FAILED,
                updated_at=datetime.now(UTC),
            )
            session.state._current = new_snapshot
            session.state._history.append(new_snapshot)

            await self.publish(
                ExecutionFailed(
                    execution_id=session.execution_id,
                    sender=self.ref,
                    data=ex,
                )
            )

            activity_bus.emit("Executor", "failed", status="failed", detail=str(ex), session_id=sid)

    async def _check_pause(self, session: 'ExecutionSession') -> None:
        """
        Check if execution should pause.

        Args:
            session: Execution session
        """
        instructions = session.get_instructions()
        if "PAUSE" in instructions:
            self._paused = True
            session.update_memory("actor_status", "paused")

            await self.publish(
                ExecutionPaused(
                    execution_id=session.execution_id,
                    sender=self.ref,
                    data=None,
                )
            )

            # Wait until resumed
            while self._paused:
                await asyncio.sleep(0.1)
                instructions = session.get_instructions()
                if "RESUME" in instructions:
                    self._paused = False
                    session.update_memory("actor_status", "resumed")

                    await self.publish(
                        ExecutionResumed(
                            execution_id=session.execution_id,
                            sender=self.ref,
                            data=None,
                        )
                    )

    async def _check_cancel(self, session: 'ExecutionSession') -> None:
        """
        Check if execution should cancel.

        Args:
            session: Execution session
        """
        instructions = session.get_instructions()
        if "CANCEL" in instructions:
            self._cancelled = True
            if self._current_task:
                self._current_task.cancel()
