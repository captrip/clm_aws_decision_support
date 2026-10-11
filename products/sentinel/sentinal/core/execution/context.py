from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from sentinal.core.capabilities import Capabilities


@dataclass(slots=True, frozen=True)
class ExecutionContext:
    """
    Immutable execution definition.

    This object describes what should be executed.
    It never changes during execution.
    """

    input: Any = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    instructions: tuple[str, ...] = ()

    capabilities: Capabilities = field(default_factory=Capabilities)

    session: Any = None
    """Live ExecutionSession when running under the runtime, else None.
    Typed Any to avoid a circular import with session.py."""