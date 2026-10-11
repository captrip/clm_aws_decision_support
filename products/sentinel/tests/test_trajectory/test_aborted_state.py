from dataclasses import replace
from datetime import UTC, datetime
from sentinal.core.execution.state import ExecutionState
from sentinal.core.execution.snapshot import ExecutionStatus
from sentinal.exceptions.runtime import ExecutionAborted, SentinelRuntimeError


def test_aborted_status_exists():
    assert ExecutionStatus.ABORTED.value == "aborted"


def test_aborted_is_terminal():
    st = ExecutionState()
    st._current = replace(st._current, status=ExecutionStatus.ABORTED,
                          updated_at=datetime.now(UTC))
    assert st.is_terminal() is True


def test_execution_aborted_is_runtime_error():
    assert issubclass(ExecutionAborted, SentinelRuntimeError)
