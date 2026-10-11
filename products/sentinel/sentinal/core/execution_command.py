from __future__ import annotations

from dataclasses import dataclass

from sentinal.core.execution import (
    ExecutionSession,
)
from sentinal.core.executable import Executable


@dataclass(slots=True, frozen=True)
class ExecutionCommand:
    """
    Complete instruction required by an Executor
    to execute a workload.
    """

    session: ExecutionSession

    executable: Executable