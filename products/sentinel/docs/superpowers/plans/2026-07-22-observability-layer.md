# Observability Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured logging and OpenTelemetry tracing to every sentinal module, with `@traced` and `@logged` decorators for package consumers.

**Architecture:** A `sentinal/telemetry/` package provides `get_tracer()` (returns real OTel tracer or no-op), structured logging setup, and two decorators. Each existing module gets a logger + tracer added. OTel is optional — everything gracefully degrades.

**Tech Stack:** Python logging, opentelemetry-api/sdk/exporter-otlp (optional), functools.wraps

## Global Constraints

- Python 3.14t with `.ft-env`, all commands: `PYTHON_GIL=0 .ft-env/bin/python`
- OTel packages are OPTIONAL — code must work without them installed
- Zero mandatory new dependencies in pyproject.toml `dependencies = []`
- OTel goes in `[project.optional-dependencies] telemetry = [...]`
- Do NOT commit anything — keep all changes local
- Existing 65 tests must continue to pass after all changes
- Default log level: INFO for `sentinal.*` loggers

---

### Task 1: Telemetry Package — Tracing Module (No-Op Safe)

**Files:**
- Create: `sentinal/telemetry/__init__.py`
- Create: `sentinal/telemetry/tracing.py`
- Test: `tests/test_telemetry/__init__.py`
- Test: `tests/test_telemetry/test_tracing.py`

**Interfaces:**
- Consumes: Nothing (foundational)
- Produces:
  - `get_tracer(name: str) -> Tracer` — returns OTel tracer or no-op
  - `NoOpSpan` context manager — used when OTel not installed
  - `NoOpTracer` class — `start_as_current_span(name, attributes=None)` returns NoOpSpan

- [ ] **Step 1: Create package structure**

```bash
mkdir -p sentinal/telemetry
mkdir -p tests/test_telemetry
touch tests/test_telemetry/__init__.py
```

- [ ] **Step 2: Write test**

```python
# tests/test_telemetry/test_tracing.py
import pytest
from unittest.mock import patch


class TestGetTracer:
    def test_returns_tracer_object(self):
        from sentinal.telemetry.tracing import get_tracer
        tracer = get_tracer("test.module")
        assert tracer is not None
        assert hasattr(tracer, "start_as_current_span")

    def test_no_op_when_otel_missing(self):
        """Tracer works even if opentelemetry is not installed."""
        import sentinal.telemetry.tracing as mod
        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
            # Force reimport logic
            from sentinal.telemetry.tracing import NoOpTracer, NoOpSpan
            tracer = NoOpTracer("test")
            span = tracer.start_as_current_span("op")
            # Works as context manager
            with span as s:
                s.set_attribute("key", "value")
                s.add_event("thing happened")

    def test_span_context_manager(self):
        from sentinal.telemetry.tracing import get_tracer
        tracer = get_tracer("test")
        with tracer.start_as_current_span("my_op", attributes={"x": 1}) as span:
            span.set_attribute("result", "ok")
        # No exception = success

    def test_span_records_exception(self):
        from sentinal.telemetry.tracing import get_tracer
        tracer = get_tracer("test")
        with pytest.raises(ValueError):
            with tracer.start_as_current_span("failing_op") as span:
                raise ValueError("test error")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_telemetry/test_tracing.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 4: Implement tracing module**

```python
# sentinal/telemetry/tracing.py
"""OpenTelemetry tracing with graceful no-op fallback."""
from __future__ import annotations

import os
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_initialized = False


def _init_otel():
    """Initialize OTel TracerProvider once."""
    global _initialized
    if _initialized or not _OTEL_AVAILABLE:
        return
    _initialized = True

    provider = TracerProvider()

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except ImportError:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)


class NoOpSpan:
    """Span that does nothing — used when OTel is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def set_status(self, status) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class NoOpTracer:
    """Tracer that does nothing — used when OTel is not installed."""

    def __init__(self, name: str = ""):
        self._name = name

    def start_as_current_span(self, name: str, attributes: dict | None = None) -> NoOpSpan:
        return NoOpSpan()


def get_tracer(name: str):
    """Get a tracer. Returns real OTel tracer if available, NoOpTracer otherwise."""
    if not _OTEL_AVAILABLE:
        return NoOpTracer(name)

    _init_otel()
    service_name = os.environ.get("OTEL_SERVICE_NAME", "sentinal")
    return trace.get_tracer(name, tracer_provider=trace.get_tracer_provider())
```

```python
# sentinal/telemetry/__init__.py
"""Sentinal observability — tracing, logging, and decorators."""
from sentinal.telemetry.tracing import get_tracer, NoOpTracer, NoOpSpan

__all__ = ["get_tracer", "NoOpTracer", "NoOpSpan"]
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_telemetry/test_tracing.py -v`
Expected: All PASS

---

### Task 2: Logging Module

**Files:**
- Create: `sentinal/telemetry/logging.py`
- Modify: `sentinal/telemetry/__init__.py`
- Test: `tests/test_telemetry/test_logging.py`

**Interfaces:**
- Consumes: Nothing
- Produces:
  - `setup_logging(level: str = "INFO", format: str = "text") -> None`
  - `JSONFormatter` class (logging.Formatter subclass)
  - Calling `setup_logging()` configures the root `sentinal` logger

- [ ] **Step 1: Write test**

```python
# tests/test_telemetry/test_logging.py
import json
import logging

import pytest

from sentinal.telemetry.logging import setup_logging, JSONFormatter


class TestSetupLogging:
    def teardown_method(self):
        # Reset sentinal logger between tests
        logger = logging.getLogger("sentinal")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    def test_sets_level_info_by_default(self):
        setup_logging()
        logger = logging.getLogger("sentinal")
        assert logger.level == logging.INFO

    def test_sets_custom_level(self):
        setup_logging(level="DEBUG")
        logger = logging.getLogger("sentinal")
        assert logger.level == logging.DEBUG

    def test_text_format_produces_readable_output(self, capsys):
        setup_logging(level="INFO", format="text")
        logger = logging.getLogger("sentinal.test")
        logger.info("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.err

    def test_json_format_produces_valid_json(self, capsys):
        setup_logging(level="INFO", format="json")
        logger = logging.getLogger("sentinal.test_json")
        logger.info("test message", extra={"backend": "cpu"})
        captured = capsys.readouterr()
        line = captured.err.strip().split("\n")[-1]
        data = json.loads(line)
        assert data["message"] == "test message"
        assert data["level"] == "INFO"
        assert "timestamp" in data

    def test_respects_env_var(self, monkeypatch):
        monkeypatch.setenv("SENTINAL_LOG_LEVEL", "WARNING")
        setup_logging()
        logger = logging.getLogger("sentinal")
        assert logger.level == logging.WARNING

    def test_child_loggers_inherit(self):
        setup_logging(level="DEBUG")
        child = logging.getLogger("sentinal.compute.runner")
        assert child.getEffectiveLevel() == logging.DEBUG


class TestJSONFormatter:
    def test_format_includes_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="sentinal.test", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "test"
        assert data["logger"] == "sentinal.test"
        assert data["level"] == "INFO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_telemetry/test_logging.py -v`
Expected: FAIL

- [ ] **Step 3: Implement logging module**

```python
# sentinal/telemetry/logging.py
"""Structured logging setup for sentinal."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, UTC


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            data["exception"] = self.formatException(record.exc_info)
        # Include any extra fields
        for key in record.__dict__:
            if key not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__ and key not in ("message", "args"):
                data[key] = record.__dict__[key]
        return json.dumps(data, default=str)


_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    level: str | None = None,
    format: str | None = None,
) -> None:
    """
    Configure sentinal loggers.

    Args:
        level: Log level (default: INFO, or SENTINAL_LOG_LEVEL env var)
        format: "text" or "json" (default: text, or SENTINAL_LOG_FORMAT env var)
    """
    level = level or os.environ.get("SENTINAL_LOG_LEVEL", "INFO")
    format = format or os.environ.get("SENTINAL_LOG_FORMAT", "text")

    logger = logging.getLogger("sentinal")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    logger.addHandler(handler)
```

- [ ] **Step 4: Update `__init__.py`**

```python
# sentinal/telemetry/__init__.py
"""Sentinal observability — tracing, logging, and decorators."""
from sentinal.telemetry.tracing import get_tracer, NoOpTracer, NoOpSpan
from sentinal.telemetry.logging import setup_logging, JSONFormatter

__all__ = [
    "get_tracer",
    "NoOpTracer",
    "NoOpSpan",
    "setup_logging",
    "JSONFormatter",
]
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_telemetry/test_logging.py -v`
Expected: All PASS

---

### Task 3: Decorators — @traced and @logged

**Files:**
- Create: `sentinal/telemetry/decorators.py`
- Modify: `sentinal/telemetry/__init__.py`
- Test: `tests/test_telemetry/test_decorators.py`

**Interfaces:**
- Consumes: `get_tracer` from Task 1, `logging` from Task 2
- Produces:
  - `traced(fn=None, *, name=None, attributes=None)` — decorator, works with/without args
  - `logged(fn=None, *, level="INFO", capture_result=False)` — decorator, works with/without args
  - Both support sync and async functions

- [ ] **Step 1: Write test**

```python
# tests/test_telemetry/test_decorators.py
import asyncio
import logging

import pytest

from sentinal.telemetry.decorators import traced, logged


class TestTracedDecorator:
    def test_sync_function(self):
        @traced
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_async_function(self):
        @traced
        async def fetch(url):
            return f"data from {url}"

        result = asyncio.run(fetch("http://example.com"))
        assert result == "data from http://example.com"

    def test_with_custom_name(self):
        @traced(name="custom.operation")
        def work():
            return "done"

        assert work() == "done"

    def test_with_attributes(self):
        @traced(attributes={"model": "gpt-4", "version": 2})
        def inference(prompt):
            return "response"

        assert inference("hello") == "response"

    def test_preserves_function_metadata(self):
        @traced
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_records_exception(self):
        @traced
        def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing()

    def test_async_exception(self):
        @traced
        async def async_fail():
            raise RuntimeError("async boom")

        with pytest.raises(RuntimeError, match="async boom"):
            asyncio.run(async_fail())


class TestLoggedDecorator:
    def test_sync_function(self, caplog):
        @logged
        def compute(x):
            return x * 2

        with caplog.at_level(logging.INFO):
            result = compute(5)
        assert result == 10
        assert any("compute" in r.message for r in caplog.records)

    def test_async_function(self, caplog):
        @logged
        async def async_compute(x):
            return x + 1

        with caplog.at_level(logging.INFO):
            result = asyncio.run(async_compute(10))
        assert result == 11

    def test_custom_level(self, caplog):
        @logged(level="DEBUG")
        def debug_fn():
            return "ok"

        with caplog.at_level(logging.DEBUG):
            debug_fn()
        assert any(r.levelno == logging.DEBUG for r in caplog.records)

    def test_capture_result(self, caplog):
        @logged(capture_result=True)
        def get_value():
            return 42

        with caplog.at_level(logging.INFO):
            get_value()
        assert any("42" in r.message for r in caplog.records)

    def test_logs_exception(self, caplog):
        @logged
        def explode():
            raise ValueError("kaboom")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                explode()
        assert any("kaboom" in r.message for r in caplog.records)

    def test_preserves_function_metadata(self):
        @logged
        def named_fn():
            """Doc here."""
            pass

        assert named_fn.__name__ == "named_fn"
        assert named_fn.__doc__ == "Doc here."


class TestComposition:
    def test_traced_and_logged_together(self, caplog):
        @traced
        @logged
        def combined(x):
            return x * 3

        with caplog.at_level(logging.INFO):
            result = combined(4)
        assert result == 12

    def test_async_composition(self, caplog):
        @traced
        @logged
        async def async_combined(x):
            return x + 10

        with caplog.at_level(logging.INFO):
            result = asyncio.run(async_combined(5))
        assert result == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_telemetry/test_decorators.py -v`
Expected: FAIL

- [ ] **Step 3: Implement decorators**

```python
# sentinal/telemetry/decorators.py
"""User-facing decorators: @traced and @logged."""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable

from sentinal.telemetry.tracing import get_tracer


def traced(fn: Callable | None = None, *, name: str | None = None, attributes: dict | None = None):
    """
    Decorator that creates an OTel span around a function call.

    Usage:
        @traced
        def my_func(): ...

        @traced(name="custom.name", attributes={"key": "val"})
        async def my_async(): ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        tracer = get_tracer(func.__module__)

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name, attributes=attributes) as span:
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except BaseException as e:
                        span.record_exception(e)
                        raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name, attributes=attributes) as span:
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except BaseException as e:
                        span.record_exception(e)
                        raise
            return sync_wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def logged(fn: Callable | None = None, *, level: str = "INFO", capture_result: bool = False):
    """
    Decorator that logs function entry, exit, and exceptions.

    Usage:
        @logged
        def my_func(): ...

        @logged(level="DEBUG", capture_result=True)
        async def my_async(): ...
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(func.__module__)
        log_level = getattr(logging, level.upper(), logging.INFO)

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger.log(log_level, f"Calling {func.__qualname__}")
                t0 = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - t0) * 1000
                    msg = f"{func.__qualname__} completed in {duration_ms:.1f}ms"
                    if capture_result:
                        msg += f" result={result}"
                    logger.log(log_level, msg)
                    return result
                except BaseException as e:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    logger.error(
                        f"{func.__qualname__} failed after {duration_ms:.1f}ms: {e}"
                    )
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger.log(log_level, f"Calling {func.__qualname__}")
                t0 = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - t0) * 1000
                    msg = f"{func.__qualname__} completed in {duration_ms:.1f}ms"
                    if capture_result:
                        msg += f" result={result}"
                    logger.log(log_level, msg)
                    return result
                except BaseException as e:
                    duration_ms = (time.perf_counter() - t0) * 1000
                    logger.error(
                        f"{func.__qualname__} failed after {duration_ms:.1f}ms: {e}"
                    )
                    raise
            return sync_wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
```

- [ ] **Step 4: Update `__init__.py`**

```python
# sentinal/telemetry/__init__.py
"""Sentinal observability — tracing, logging, and decorators."""
from sentinal.telemetry.tracing import get_tracer, NoOpTracer, NoOpSpan
from sentinal.telemetry.logging import setup_logging, JSONFormatter
from sentinal.telemetry.decorators import traced, logged

__all__ = [
    "get_tracer",
    "NoOpTracer",
    "NoOpSpan",
    "setup_logging",
    "JSONFormatter",
    "traced",
    "logged",
]
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_telemetry/test_decorators.py -v`
Expected: All PASS

---

### Task 4: `configure()` Function

**Files:**
- Modify: `sentinal/telemetry/__init__.py`
- Test: `tests/test_telemetry/test_configure.py`

**Interfaces:**
- Consumes: `setup_logging` (Task 2), `_init_otel` from tracing (Task 1)
- Produces: `configure(service_name="sentinal", log_level="INFO", log_format="text", trace_exporter="auto") -> None`

- [ ] **Step 1: Write test**

```python
# tests/test_telemetry/test_configure.py
import logging
import os

import pytest

from sentinal.telemetry import configure


class TestConfigure:
    def teardown_method(self):
        logger = logging.getLogger("sentinal")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    def test_configures_logging(self):
        configure(log_level="DEBUG")
        logger = logging.getLogger("sentinal")
        assert logger.level == logging.DEBUG

    def test_configures_json_format(self, capsys):
        configure(log_format="json")
        logger = logging.getLogger("sentinal.test_cfg")
        logger.info("json test")
        captured = capsys.readouterr()
        assert '"message": "json test"' in captured.err

    def test_default_values(self):
        configure()
        logger = logging.getLogger("sentinal")
        assert logger.level == logging.INFO

    def test_disable_tracing(self):
        # Should not raise even if OTel not installed
        configure(trace_exporter="none")

    def test_service_name(self, monkeypatch):
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        configure(service_name="my-agent")
        # Verify it doesn't crash; actual OTel verification needs SDK
```

- [ ] **Step 2: Implement configure()**

Add to `sentinal/telemetry/__init__.py`:

```python
# sentinal/telemetry/__init__.py
"""Sentinal observability — tracing, logging, and decorators."""
import os

from sentinal.telemetry.tracing import get_tracer, NoOpTracer, NoOpSpan
from sentinal.telemetry.logging import setup_logging, JSONFormatter
from sentinal.telemetry.decorators import traced, logged


def configure(
    service_name: str = "sentinal",
    log_level: str = "INFO",
    log_format: str = "text",
    trace_exporter: str = "auto",
) -> None:
    """
    Configure sentinal observability.

    Args:
        service_name: OTel service name
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: "text" or "json"
        trace_exporter: "auto", "console", "otlp", or "none"
    """
    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)

    if trace_exporter == "none":
        os.environ["SENTINAL_TRACING_DISABLED"] = "1"
    elif trace_exporter == "otlp":
        # User should also set OTEL_EXPORTER_OTLP_ENDPOINT
        pass
    elif trace_exporter == "console":
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)

    setup_logging(level=log_level, format=log_format)


__all__ = [
    "get_tracer",
    "NoOpTracer",
    "NoOpSpan",
    "setup_logging",
    "JSONFormatter",
    "traced",
    "logged",
    "configure",
]
```

Also update `sentinal/telemetry/tracing.py` to respect `SENTINAL_TRACING_DISABLED`:

In `get_tracer()`, add at the top:
```python
if os.environ.get("SENTINAL_TRACING_DISABLED") == "1":
    return NoOpTracer(name)
```

- [ ] **Step 3: Run tests**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_telemetry/ -v`
Expected: All PASS

---

### Task 5: Instrument Existing Modules

**Files:**
- Modify: `sentinal/runtime/runtime.py`
- Modify: `sentinal/compute/runner.py`
- Modify: `sentinal/compute/profiler.py`
- Modify: `sentinal/compute/cpu_backend.py`
- Modify: `sentinal/compute/mps_backend.py`
- Modify: `sentinal/compute/metal_backend.py`
- Modify: `sentinal/actors/executor.py`
- Modify: `sentinal/actors/observer.py`
- Modify: `sentinal/actors/governor.py`
- Test: `tests/test_telemetry/test_instrumentation.py`

**Interfaces:**
- Consumes: `get_tracer` (Task 1), `logging` (stdlib)
- Produces: Each module emits spans and logs at appropriate levels

- [ ] **Step 1: Write test that verifies instrumentation produces logs**

```python
# tests/test_telemetry/test_instrumentation.py
import logging

import pytest

from sentinal.compute.device import DevicePreference
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.runtime.runtime import SentinelRuntime
from sentinal.telemetry import configure


class SimpleExecutable(Executable):
    device = DevicePreference.CPU

    async def execute(self, context):
        return {"status": "done"}


class TestInstrumentation:
    @pytest.fixture(autouse=True)
    def setup_logging(self):
        configure(log_level="DEBUG", trace_exporter="none")

    @pytest.mark.asyncio
    async def test_runtime_logs_execution(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="sentinal"):
            runtime = SentinelRuntime()
            session = ExecutionSession(
                executable=SimpleExecutable(),
                context=ExecutionContext(input={}),
            )
            await runtime.execute(session)
            await runtime.shutdown()

        log_messages = " ".join(r.message for r in caplog.records)
        # Runtime should log session start
        assert "execute" in log_messages.lower() or "session" in log_messages.lower()

    @pytest.mark.asyncio
    async def test_runner_logs_backend_dispatch(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="sentinal"):
            runtime = SentinelRuntime()
            session = ExecutionSession(
                executable=SimpleExecutable(),
                context=ExecutionContext(input={}),
            )
            await runtime.execute(session)
            await runtime.shutdown()

        loggers_used = set(r.name for r in caplog.records)
        # At least runtime and runner should have logged
        assert any("runtime" in name for name in loggers_used)

    @pytest.mark.asyncio
    async def test_profiler_logs_resolution(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="sentinal"):
            runtime = SentinelRuntime()
            session = ExecutionSession(
                executable=SimpleExecutable(),
                context=ExecutionContext(input={}),
            )
            await runtime.execute(session)
            await runtime.shutdown()

        log_messages = " ".join(r.message for r in caplog.records)
        assert "cpu" in log_messages.lower()
```

- [ ] **Step 2: Add logging and tracing to `sentinal/runtime/runtime.py`**

Add at the top (after existing imports):
```python
import logging
from sentinal.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
```

In `execute()`, wrap the body:
```python
async def execute(self, session):
    with tracer.start_as_current_span("sentinal.runtime.execute", attributes={
        "session.id": str(session.session_id),
        "executable.type": type(session.executable).__name__,
    }) as span:
        logger.info(f"Executing session {session.session_id}")
        # ... existing code ...
        logger.info(f"Session {session.session_id} completed")
        return result
```

In `shutdown()`:
```python
logger.info("Runtime shutting down")
```

- [ ] **Step 3: Add logging and tracing to `sentinal/compute/runner.py`**

Add at top:
```python
import logging
from sentinal.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
```

In `run_actors()`:
```python
logger.info(f"Dispatching actors: executor={backend_name}, observer=cpu, governor=cpu")
```

In `detect_backends()`:
```python
logger.info(f"Detected backends: {[b.name for b in backends]}")
```

- [ ] **Step 4: Add logging to `sentinal/compute/profiler.py`**

Add at top:
```python
import logging

logger = logging.getLogger(__name__)
```

In `resolve()`, after caching:
```python
logger.debug(f"Resolved {type(executable).__name__} -> {result}")
```

- [ ] **Step 5: Add logging to backends**

For each of `cpu_backend.py`, `mps_backend.py`, `metal_backend.py`, add:
```python
import logging
logger = logging.getLogger(__name__)
```

And a `logger.debug(...)` in the `dispatch()` method.

- [ ] **Step 6: Add logging to actors**

For `executor.py`, `observer.py`, `governor.py`, add:
```python
import logging
logger = logging.getLogger(__name__)
```

Replace the `print()` statements with proper logging:
- `print(f"👁️  [Observer...]")` → `logger.info("Observer started")`
- `print(f"👑 [Governor...]")` → `logger.info("Governor started")`

- [ ] **Step 7: Run instrumentation tests**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_telemetry/test_instrumentation.py -v -s`
Expected: All PASS

- [ ] **Step 8: Run full test suite to verify no regressions**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/ -v --tb=short -q`
Expected: All tests pass (65 existing + new telemetry tests)

---

### Task 6: Update Package Exports and Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `sentinal/__init__.py`
- Test: `tests/test_telemetry/test_package.py`

**Interfaces:**
- Consumes: All telemetry modules (Tasks 1-5)
- Produces: Updated package with telemetry optional dep, top-level exports for `traced`, `logged`, `configure`

- [ ] **Step 1: Write test**

```python
# tests/test_telemetry/test_package.py
def test_top_level_imports():
    """Package consumers can import decorators from top level."""
    from sentinal.telemetry import traced, logged, configure, get_tracer
    assert callable(traced)
    assert callable(logged)
    assert callable(configure)
    assert callable(get_tracer)


def test_sentinal_package_exports_telemetry():
    """sentinal package exposes telemetry at top level."""
    import sentinal
    assert hasattr(sentinal, "traced")
    assert hasattr(sentinal, "logged")
    assert hasattr(sentinal, "configure")
```

- [ ] **Step 2: Update `sentinal/__init__.py`**

```python
# sentinal/__init__.py
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
from sentinal.telemetry import traced, logged, configure

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
]
```

- [ ] **Step 3: Update `pyproject.toml`**

Add `telemetry` extra and update `all`:

```toml
[project.optional-dependencies]
gpu = ["torch>=2.0"]
metal = ["pyobjc-framework-Metal>=9.0"]
telemetry = [
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "opentelemetry-exporter-otlp>=1.20",
]
all = [
    "torch>=2.0",
    "pyobjc-framework-Metal>=9.0",
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "opentelemetry-exporter-otlp>=1.20",
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
]
```

- [ ] **Step 4: Reinstall package and run all tests**

Run: `PYTHON_GIL=0 .ft-env/bin/pip install -e ".[all,dev]" && PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/ -v --tb=short -q`
Expected: All PASS

---

## Task Dependency Graph

```
Task 1 (Tracing/NoOp)
├── Task 2 (Logging) ← independent of 1
├── Task 3 (Decorators) ← depends on 1, 2
│   └── Task 4 (configure()) ← depends on 1, 2
│       └── Task 5 (Instrument modules) ← depends on 1, 2, 4
│           └── Task 6 (Package exports) ← depends on all
```

Tasks 1 and 2 can run in parallel. Tasks 3-6 are sequential.