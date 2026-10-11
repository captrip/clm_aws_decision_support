# Observability Layer — Design Spec

## Summary

Add structured logging and OpenTelemetry tracing to the sentinal package. Every module emits traces and logs. Two user-facing decorators (`@traced`, `@logged`) allow consumers to extend observability to their own code with zero boilerplate.

## Goals

1. Every significant operation produces an OTel span with relevant attributes
2. Structured logging at INFO level by default across all modules
3. Configurable via `configure()` call or environment variables (zero-code)
4. Graceful degradation — decorators work without OTel SDK installed (logging still works, tracing becomes no-op)
5. Two composable decorators for package consumers: `@traced` and `@logged`
6. Zero mandatory new dependencies (OTel is optional)

## Non-Goals

- Metrics/counters (future work)
- Distributed tracing across network boundaries
- Log aggregation infrastructure

## Architecture

```
sentinal/telemetry/
├── __init__.py          # Public API: traced, logged, configure
├── tracing.py           # OTel tracer setup, get_tracer(), span helpers
├── logging.py           # Structured logger factory, JSON formatter
└── decorators.py        # @traced and @logged implementations
```

## Components

### 1. Tracing (`sentinal/telemetry/tracing.py`)

Provides `get_tracer(name: str)` that returns either a real OTel tracer or a no-op wrapper.

```python
def get_tracer(name: str) -> Tracer:
    """Get a tracer. Returns no-op if OTel SDK not installed."""
    ...
```

Tracer initialization is lazy — first call to `get_tracer()` triggers setup based on environment:
- If `opentelemetry-sdk` is importable: create a real TracerProvider
- If `OTEL_EXPORTER_OTLP_ENDPOINT` is set: use OTLP exporter
- Otherwise: use ConsoleSpanExporter (prints to stderr)
- If OTel not installed: return a no-op tracer that does nothing

### 2. Logging (`sentinal/telemetry/logging.py`)

Provides `setup_logging()` and a JSON formatter.

```python
def setup_logging(level: str = "INFO", format: str = "text") -> None:
    """Configure sentinal loggers."""
    ...
```

All sentinal modules use `logging.getLogger(__name__)`. This function configures the root `sentinal` logger with the appropriate handler and formatter.

Log levels by module:
| Level | What gets logged |
|-------|-----------------|
| INFO | Backend detected, device routed, session started/completed, profiler decision cached |
| WARNING | GPU unavailable (fallback), profiler benchmark failed |
| ERROR | Shader compilation failed, actor exception, dispatch failure |
| DEBUG | Observer poll tick, governor check, individual dispatch calls, span details |

### 3. Decorators (`sentinal/telemetry/decorators.py`)

**`@traced`** — creates an OTel span:
```python
@traced
async def my_step(ctx):
    ...

@traced(name="custom.name", attributes={"key": "value"})
def another_step():
    ...
```

Behavior:
- Span name defaults to `"module.function_name"`
- Records exceptions as span events with status ERROR
- Sets span attributes: function name, module, any user-provided attributes
- Works with both sync and async functions
- No-op if OTel not installed (function runs normally, no overhead)

**`@logged`** — emits structured log entry/exit:
```python
@logged
def process(data):
    ...

@logged(level="DEBUG", capture_result=True)
async def fetch(url):
    ...
```

Behavior:
- Logs at entry: function name, args (if capture_args=True)
- Logs at exit: function name, duration_ms, result (if capture_result=True)
- Logs exceptions at ERROR level with traceback
- Works with both sync and async functions
- Always works (no optional dependency needed)

Both decorators are composable: `@traced @logged def foo(): ...`

### 4. Configuration (`sentinal/telemetry/__init__.py`)

```python
def configure(
    service_name: str = "sentinal",
    log_level: str = "INFO",
    log_format: str = "text",       # "text" or "json"
    trace_exporter: str = "auto",   # "auto", "console", "otlp", or "none"
) -> None:
    """
    Configure sentinal observability.

    Can also be configured via environment variables:
    - OTEL_SERVICE_NAME: service name (default: "sentinal")
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector URL
    - SENTINAL_LOG_LEVEL: log level override
    - SENTINAL_LOG_FORMAT: "json" or "text"
    """
```

`trace_exporter="auto"` means:
- If `OTEL_EXPORTER_OTLP_ENDPOINT` is set → OTLP exporter
- Else → ConsoleSpanExporter (prints spans to stderr)
- `"none"` disables tracing entirely

## Span Hierarchy

```
sentinal.runtime.execute [session.id, executable.type]
├── sentinal.compute.profiler.resolve [device.preference, result]
├── sentinal.compute.runner.run_actors [backend.name]
│   ├── sentinal.compute.runner.dispatch_executor [backend, thread.id]
│   ├── sentinal.compute.runner.dispatch_observer [thread.id]
│   └── sentinal.compute.runner.dispatch_governor [thread.id]
└── sentinal.actors.executor.execute_session [status, duration_ms]
    └── user.code (via @traced decorator)
```

## Integration into Existing Modules

Each module gets two lines at the top:
```python
import logging
from sentinal.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
```

Key operations wrap in spans:
```python
with tracer.start_as_current_span("operation_name", attributes={...}) as span:
    logger.info("Starting operation", extra={...})
    # ... work ...
    span.set_attribute("result", value)
```

### Modules to instrument:
- `sentinal/runtime/runtime.py` — execute(), start(), stop(), shutdown()
- `sentinal/compute/runner.py` — run_actors(), each dispatch helper
- `sentinal/compute/profiler.py` — resolve(), profile_and_cache()
- `sentinal/compute/cpu_backend.py` — dispatch()
- `sentinal/compute/mps_backend.py` — dispatch()
- `sentinal/compute/metal_backend.py` — dispatch(), _dispatch_kernel()
- `sentinal/actors/executor.py` — execute_session()
- `sentinal/actors/observer.py` — observe_session()
- `sentinal/actors/governor.py` — govern_session()

## Dependencies

```toml
[project.optional-dependencies]
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
```

The core package has zero mandatory deps. OTel is optional — decorators and internal instrumentation gracefully no-op without it.

## Testing Strategy

1. **Unit tests for decorators**: Verify @traced and @logged work with sync/async, capture exceptions, produce logs
2. **No-op test**: Verify everything works when opentelemetry is NOT installed (mock the import failure)
3. **Integration test**: With OTel SDK, verify spans are produced with correct hierarchy and attributes
4. **Logging test**: Verify structured output at correct levels, JSON format option
5. **Configuration test**: Verify env var and explicit configure() both work

## Error Handling

- OTel import fails → no-op tracer, log a DEBUG message about it
- OTLP endpoint unreachable → OTel SDK handles this (queues/drops, doesn't crash)
- Decorator on a non-callable → raise TypeError at decoration time
- configure() called multiple times → last call wins, warn at WARNING level