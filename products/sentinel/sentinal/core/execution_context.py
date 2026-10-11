from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import UUID, uuid4

from .capabilities import Capabilities


@dataclass(slots=True, frozen=True)
class ExecutionContext:
    """
    Immutable execution context supplied to an Executable.

    This object never changes during execution.

    It contains only information that defines *what* is being
    executed—not the execution's evolving state.

    Mutable runtime data belongs in ExecutionState.
    """

    execution_id: UUID = field(default_factory=uuid4)

    session_id: UUID = field(default_factory=uuid4)

    input: Any = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    instructions: tuple[str, ...] = ()

    capabilities: Capabilities = field(default_factory=Capabilities)