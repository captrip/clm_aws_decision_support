from enum import StrEnum


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(StrEnum):
    EXECUTION_STARTED = "execution.started"
    EXECUTION_FINISHED = "execution.finished"

    BRAIN_STARTED = "brain.started"
    BRAIN_FINISHED = "brain.finished"

    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"

    SUMMARY_STARTED = "summary.started"
    SUMMARY_FINISHED = "summary.finished"

    ERROR = "error"


class ObservationSeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SignalType(StrEnum):
    CONTINUE = "continue"
    RETRY = "retry"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    ESCALATE = "escalate"
    MODIFY_CONTEXT = "modify_context"