"""
SkillExecutor Actor - Sentinel actor that executes individual skills.

Replaces the concurrent asyncio.TaskGroup approach with Sentinel's actor model,
enabling built-in retry policies via Governor and better observability.
"""

from __future__ import annotations

from typing import Any, Dict
from dataclasses import dataclass

from sentinal.actors.actor import BaseActor
from sentinal.core.contracts.execution_events import (
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionStarted,
)
from sentinal.core.contracts.execution_signals import StartExecution
from sentinal.core.execution.context import ExecutionContext

from MAF.integration.executables import SkillExecutable


@dataclass
class ExecuteSkillSignal:
    """
    Signal to execute a skill with given state.
    
    Attributes:
        skill_name: Name of the skill to execute
        skill_entry: Dict containing runner/graph + metadata
        state: Input state for skill execution
        execution_id: UUID for tracking
    """
    skill_name: str
    skill_entry: Dict[str, Any]
    state: Dict[str, Any]
    execution_id: str


class SkillExecutor(BaseActor):
    """
    Actor that executes individual skills with Sentinel integration.
    
    Receives ExecuteSkillSignal or StartExecution messages, invokes the skill
    via SkillExecutable, and publishes completion/failure events for Governor
    to observe and retry if needed.
    
    Replaces the direct asyncio.gather() approach in dispatcher_concurrent.py,
    allowing:
    - Governor-driven retry policies
    - Better observability via events
    - Distributed execution (actors can be spread across processes)
    - Natural backpressure via mailbox
    """
    
    async def receive(self, message: Any) -> None:
        """
        Handle incoming messages.
        
        Supports both:
        - ExecuteSkillSignal: Custom skill execution signal
        - StartExecution: Generic Sentinel execution signal
        """
        if isinstance(message, ExecuteSkillSignal):
            await self._execute_skill(message)
        elif isinstance(message, StartExecution):
            await self._execute_via_start_signal(message)
    
    async def _execute_skill(self, signal: ExecuteSkillSignal) -> None:
        """Execute skill via ExecuteSkillSignal."""
        skill_name = signal.skill_name
        skill_entry = signal.skill_entry
        state = signal.state
        execution_id = signal.execution_id
        
        try:
            # Create SkillExecutable
            executable = SkillExecutable(
                skill_name=skill_name,
                runner=skill_entry.get("runner"),
                graph=skill_entry.get("graph"),
                state_manager=skill_entry.get("state_mgr"),
                scope=skill_entry.get("scope", skill_name),
            )
            
            # Create ExecutionContext for the skill
            context = ExecutionContext(
                input=state,
                metadata={
                    "skill_name": skill_name,
                    "execution_id": str(execution_id),
                },
            )
            
            # Execute
            result = await executable.execute(context)
            
            # Publish success event
            await self.publish(
                ExecutionCompleted(
                    execution_id=execution_id,
                    sender=self.ref,
                    data=result,
                )
            )
        
        except Exception as ex:
            # Publish failure event (Governor will observe)
            await self.publish(
                ExecutionFailed(
                    execution_id=execution_id,
                    sender=self.ref,
                    data={
                        "skill_name": skill_name,
                        "error": str(ex),
                    },
                )
            )
    
    async def _execute_via_start_signal(self, message: StartExecution) -> None:
        """Execute skill via standard Sentinel StartExecution signal."""
        command = message.data
        session = command.session
        executable = command.executable
        
        started = ExecutionStarted(
            execution_id=session.execution_id,
            sender=self.ref,
            data=None,
        )
        session.state.apply(started)
        await self.publish(started)
        
        try:
            result = await executable.execute(session.context)
            
            completed = ExecutionCompleted(
                execution_id=session.execution_id,
                sender=self.ref,
                data=result,
            )
            session.state.apply(completed)
            await self.publish(completed)
        
        except Exception as ex:
            failed = ExecutionFailed(
                execution_id=session.execution_id,
                sender=self.ref,
                data=ex,
            )
            session.state.apply(failed)
            await self.publish(failed)
