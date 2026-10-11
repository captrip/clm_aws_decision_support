class SentinelRuntimeError(Exception):
    """Base Sentinel exception."""


class ExecutionCancelled(SentinelRuntimeError):
    pass


class ExecutionAborted(SentinelRuntimeError):
    """Raised when the Governor stops a run mid-flight (STOP correction)."""
    pass


class RetryLimitExceeded(SentinelRuntimeError):
    pass


class AdapterError(SentinelRuntimeError):
    pass