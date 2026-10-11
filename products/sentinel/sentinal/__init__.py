"""Sentinal — Hybrid compute runtime for multi-threaded agent supervision."""

from sentinal.compute import (
    ComputeBackend,
    CPUBackend,
    DevicePreference,
    HybridRunner,
    MetalBackend,
    MetalKernel,
    MPSBackend,
    WorkloadProfiler,
)
from sentinal.telemetry import traced, logged, configure, activity_bus, LiveThreadDashboard
from sentinal.core.trajectory import (
    SEVERITY,
    Correction,
    CorrectionFlag,
    JudgeContext,
    TrajectoryJudge,
    TrajectoryStep,
)
from sentinal.judges import (
    FunctionJudge,
    HeuristicJudge,
    LatencyBudgetJudge,
    OutputLoopJudge,
    RepeatToolJudge,
    StepBudgetJudge,
    avoid_tool,
    finalize,
    judge,
    redirect,
    stop,
)
from sentinal.integrations.mona import MonaAPIError, MonaClient

__version__ = "0.1.0"

__all__ = [
    "ComputeBackend",
    "CPUBackend",
    "DevicePreference",
    "HybridRunner",
    "MetalBackend",
    "MetalKernel",
    "MPSBackend",
    "WorkloadProfiler",
    "traced",
    "logged",
    "configure",
    "activity_bus",
    "LiveThreadDashboard",
    "CorrectionFlag",
    "Correction",
    "TrajectoryStep",
    "TrajectoryJudge",
    "JudgeContext",
    "SEVERITY",
    "judge",
    "FunctionJudge",
    "redirect",
    "avoid_tool",
    "finalize",
    "stop",
    "StepBudgetJudge",
    "RepeatToolJudge",
    "OutputLoopJudge",
    "LatencyBudgetJudge",
    "HeuristicJudge",
    "MonaAPIError",
    "MonaClient",
]