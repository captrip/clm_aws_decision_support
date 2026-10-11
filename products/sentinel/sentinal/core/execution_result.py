from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExecutionResult:

    output: Any = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )