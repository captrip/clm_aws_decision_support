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
    if os.environ.get("SENTINAL_TRACING_DISABLED") == "1":
        return NoOpTracer(name)

    if not _OTEL_AVAILABLE:
        return NoOpTracer(name)

    _init_otel()
    return trace.get_tracer(name, tracer_provider=trace.get_tracer_provider())