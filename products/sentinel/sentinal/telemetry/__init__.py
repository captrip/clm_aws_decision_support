"""Sentinal observability — tracing, logging, and decorators."""
import os

from sentinal.telemetry.tracing import get_tracer, NoOpTracer, NoOpSpan
from sentinal.telemetry.logging import setup_logging, JSONFormatter
from sentinal.telemetry.decorators import traced, logged
from sentinal.telemetry.dashboard import (
    activity_bus,
    LiveThreadDashboard,
    dashboard_enabled,
)


def configure(
    service_name: str = "sentinal",
    log_level: str = "INFO",
    log_format: str = "text",
    trace_exporter: str = "auto",
    live_dashboard: bool = False,
) -> None:
    """
    Configure sentinal observability.

    Args:
        service_name: OTel service name (default: "sentinal")
        log_level: Log level — DEBUG, INFO, WARNING, ERROR (default: INFO)
        log_format: "text" or "json" (default: text)
        trace_exporter: "auto", "console", "otlp", or "none" (default: auto)
            - "auto": OTLP if OTEL_EXPORTER_OTLP_ENDPOINT set, else console
            - "none": disables tracing entirely
        live_dashboard: Enable live thread dashboard (default: False)
    """
    if live_dashboard:
        os.environ["SENTINAL_LIVE_DASHBOARD"] = "1"

    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)

    if trace_exporter == "none":
        os.environ["SENTINAL_TRACING_DISABLED"] = "1"
    elif trace_exporter == "otlp":
        pass  # User should set OTEL_EXPORTER_OTLP_ENDPOINT
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
    "activity_bus",
    "LiveThreadDashboard",
    "dashboard_enabled",
]
