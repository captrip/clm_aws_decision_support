"""Trajectory steps, judge input, and Governor correction flags for mid-flight steering."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class CorrectionFlag(str, Enum):
    CONTINUE = "continue"            # all good, no-op (never pushed)
    REDIRECT = "redirect"            # steer: use note as guidance
    AVOID_TOOL = "avoid_tool"        # note = tool name to stop calling
    FORCE_FINALIZE = "force_finalize"  # wrap up now
    STOP = "stop"                    # abort the run


# How escalatory each flag is. When several judges disagree on one step, the
# Governor pushes the highest-severity correction and records the rest.
SEVERITY: dict[CorrectionFlag, int] = {
    CorrectionFlag.CONTINUE: 0,
    CorrectionFlag.REDIRECT: 1,
    CorrectionFlag.AVOID_TOOL: 2,
    CorrectionFlag.FORCE_FINALIZE: 3,
    CorrectionFlag.STOP: 4,
}


@dataclass(frozen=True)
class TrajectoryStep:
    index: int
    kind: str          # "agent" | "tool" | "message"
    summary: str       # short human-readable
    raw: dict          # streamed chunk, for the judge
    ts: float = 0.0


@dataclass(frozen=True)
class Correction:
    flag: CorrectionFlag
    note: str | None = None
    step_index: int = -1
    ts: float = 0.0
    meta: Mapping[str, Any] = field(default_factory=dict)
    """Free-form structured detail from the judge (e.g. {"spend": 2.14}).
    Never interpreted by the framework; carried to the interventions log and
    the dashboard so business metrics survive to the operator."""

    def __post_init__(self):
        # Freeze meta so callers cannot mutate it and prevent external aliasing.
        object.__setattr__(self, "meta", _read_only(self.meta))


def _read_only(m: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(m))


@dataclass(frozen=True)
class JudgeContext:
    """What a judge sees. Sequence-like over `trajectory`, so judges written
    against `list[TrajectoryStep]` keep working unchanged: len(), indexing,
    slicing and iteration all delegate to the trajectory."""

    trajectory: Sequence[TrajectoryStep]
    """Frozen snapshot of the trajectory. A tuple, so a judge cannot mutate what
    its peers see — while len(), indexing, slicing and iteration all still work
    for judges written against a plain list."""
    elapsed_s: float = 0.0
    """Wall-clock seconds since the Governor started governing this session."""
    memory: Mapping[str, Any] = field(default_factory=dict)
    """Read-only snapshot of session memory."""
    input: Mapping[str, Any] = field(default_factory=dict)
    """The execution input, e.g. {"query": "..."}."""

    def __post_init__(self):
        # Freeze the trajectory and the mappings so one judge cannot mutate
        # session state or blind its peers.
        object.__setattr__(self, "trajectory", tuple(self.trajectory))
        object.__setattr__(self, "memory", _read_only(self.memory))
        object.__setattr__(self, "input", _read_only(self.input))

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(
        self, i: int | slice
    ) -> TrajectoryStep | tuple[TrajectoryStep, ...]:
        return self.trajectory[i]

    def __iter__(self):
        return iter(self.trajectory)


@runtime_checkable
class TrajectoryJudge(Protocol):
    def judge(self, ctx: JudgeContext) -> Correction | None: ...
