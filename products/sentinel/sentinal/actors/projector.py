from __future__ import annotations

from sentinal.actors.actor import BaseActor
from sentinal.actors.decorators import handles

from sentinal.core.contracts.execution_events import (
    ExecutionStarted,
    ExecutionCompleted,
    ExecutionFailed,
)

from sentinal.runtime.session_registry import SessionRegistry


class Projector(BaseActor):
    """
    Projects execution events onto ExecutionState.

    This is the only actor that mutates ExecutionState.
    """

    def __init__(
        self,
        ref,
        registry: SessionRegistry,
    ) -> None:

        super().__init__(ref)

        self._registry = registry

    @handles(ExecutionStarted)
    async def on_started(
        self,
        event: ExecutionStarted,
    ) -> None:

        session = self._registry.get(
            event.execution_id
        )

        session.state.apply(event)

    @handles(ExecutionCompleted)
    async def on_completed(
        self,
        event: ExecutionCompleted,
    ) -> None:

        session = self._registry.get(
            event.execution_id
        )

        session.state.apply(event)

    @handles(ExecutionFailed)
    async def on_failed(
        self,
        event: ExecutionFailed,
    ) -> None:

        session = self._registry.get(
            event.execution_id
        )

        session.state.apply(event)