import pytest


class TestGetTracer:
    def test_returns_tracer_object(self):
        from sentinal.telemetry.tracing import get_tracer
        tracer = get_tracer("test.module")
        assert tracer is not None
        assert hasattr(tracer, "start_as_current_span")

    def test_no_op_tracer_works(self):
        from sentinal.telemetry.tracing import NoOpTracer, NoOpSpan
        tracer = NoOpTracer("test")
        span = tracer.start_as_current_span("op")
        with span as s:
            s.set_attribute("key", "value")
            s.add_event("thing happened")

    def test_span_context_manager(self):
        from sentinal.telemetry.tracing import get_tracer
        tracer = get_tracer("test")
        with tracer.start_as_current_span("my_op", attributes={"x": 1}) as span:
            span.set_attribute("result", "ok")

    def test_span_does_not_swallow_exceptions(self):
        from sentinal.telemetry.tracing import get_tracer
        tracer = get_tracer("test")
        with pytest.raises(ValueError):
            with tracer.start_as_current_span("failing_op") as span:
                raise ValueError("test error")

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("SENTINAL_TRACING_DISABLED", "1")
        from sentinal.telemetry.tracing import get_tracer, NoOpTracer
        tracer = get_tracer("test")
        assert isinstance(tracer, NoOpTracer)