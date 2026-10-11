"""
SentinelRuntime - Orchestrates Actor, Observer, Governor

Manages:
- Actor lifecycle (start/stop)
- Session registry
- Concurrent execution
- Complete 3-thread supervision
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional
from uuid import UUID

from sentinal.compute.runner import HybridRunner
from sentinal.core.actor_ref import ActorRef
from sentinal.core.contracts.execution_signals import StartExecution
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.messaging.dispatcher import DefaultDispatcher
from sentinal.runtime.session_registry import SessionRegistry
from sentinal.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class SentinelRuntime:
    """
    Sentinel runtime - orchestrates everything.

    Creates and manages:
    - Executor (Actor thread)
    - Observer (Observer thread)
    - Governor (Governor thread)
    - Session registry
    - Message dispatcher
    """

    def __init__(self, quality_threshold: float = 0.75):
        """
        Initialize runtime.

        Args:
            quality_threshold: Quality threshold for Governor
        """
        self.dispatcher = DefaultDispatcher()
        self.registry = SessionRegistry()
        self.actors = []
        self.quality_threshold = quality_threshold
        self._runner = HybridRunner()

        # Runtime ref
        self.ref = ActorRef("runtime", "local")

        # Track running
        self._running = False

    def register(self, actor) -> None:
        """
        Register an actor.

        Args:
            actor: Actor to register
        """
        self.actors.append(actor)

        # Attach dispatcher to actor
        actor.dispatcher = self.dispatcher

        # Register with dispatcher
        self.dispatcher.register(
            actor.ref,
            actor.mailbox,
        )

    async def start(self) -> None:
        """Start all actors."""
        if self._running:
            return

        self._running = True

        for actor in self.actors:
            await actor.start()

    async def stop(self) -> None:
        """Stop all actors."""
        if not self._running:
            return

        self._running = False

        for actor in self.actors:
            await actor.stop()

    async def submit(
        self,
        session: ExecutionSession,
    ) -> None:
        """
        Submit session for execution (message-based).

        This is the message-based approach using the dispatcher.

        Args:
            session: Session to execute
        """
        # Add to registry
        self.registry.add(session)

        # Publish StartExecution signal
        await self.dispatcher.publish(
            StartExecution(
                execution_id=session.execution_id,
                sender=self.ref,
                data=session.execution_id,
            )
        )

    async def execute(
        self,
        session: ExecutionSession,
        judges=None,
    ) -> ExecutionSession:
        """
        Execute session with 3-thread supervision (direct).

        This is the direct approach without message passing.
        Runs Actor, Observer, Governor concurrently.

        Args:
            session: Session to execute
            judges: Optional judges for THIS session only. Overrides the judges
                of a pre-registered Governor for this run, so callers no longer
                have to register one before calling execute(). The override is
                scoped to the session, so it neither leaks into a later
                execute() nor races a concurrent one.

        Returns:
            Completed session
        """
        logger.info(f"Executing session {session.session_id}")

        with tracer.start_as_current_span(
            "sentinal.runtime.execute",
            attributes={
                "session.id": str(session.session_id),
                "executable.type": type(session.executable).__name__
            }
        ) as span:
            # Add to registry
            self.registry.add(session)

            # Get or create actors
            executor = self._get_actor("executor")
            observer = self._get_actor("observer")
            governor = self._get_actor("governor")

            # Import here to avoid circular dependency
            from sentinal.actors.executor import Executor
            from sentinal.actors.observer import Observer
            from sentinal.actors.governor import Governor

            # Create if needed
            if not executor:
                executor = Executor(ActorRef("executor", "local"))
                self.register(executor)

            if not observer:
                observer = Observer(ActorRef("observer", "local"))
                self.register(observer)

            if not governor:
                governor = Governor(
                    ActorRef("governor", "local"),
                    quality_threshold=self.quality_threshold,
                )
                self.register(governor)

            # The Governor is per-runtime but judges= is per-call, so bind the
            # judges to THIS session rather than mutating shared actor state.
            # Must happen before run_actors(), which calls govern_session()
            # without a judges argument.
            if judges is not None:
                logger.info(
                    "Governing session %s with the judges= passed to execute()",
                    session.session_id)
                governor.set_session_judges(session.session_id, judges)

            # Start actors if not running
            if not self._running:
                await self.start()

            # Run all 3 threads concurrently via HybridRunner
            result = await self._runner.run_actors(session, executor, observer, governor)

            logger.info(f"Session {session.session_id} completed")
            return result

    def _get_actor(self, name: str):
        """Get actor by name."""
        for actor in self.actors:
            if actor.ref.name == name:
                return actor
        return None

    async def get_session(self, execution_id: UUID) -> Optional[ExecutionSession]:
        """
        Get session by execution ID.

        Args:
            execution_id: Execution ID

        Returns:
            Session or None
        """
        return self.registry.get(execution_id)

    def get_metrics(self) -> Dict:
        """Get runtime metrics."""
        return {
            "actors": len(self.actors),
            "running": self._running,
            "sessions": len(self.registry._sessions),
            "backends": self._runner.available_backends,
        }

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Runtime shutting down")
        await self.stop()
        self._runner.shutdown()
        self.actors.clear()
        self.registry.clear()
