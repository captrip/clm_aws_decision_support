import logging
import os

import pytest

from sentinal.telemetry import configure


class TestConfigure:
    def teardown_method(self):
        logger = logging.getLogger("sentinal")
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        os.environ.pop("SENTINAL_TRACING_DISABLED", None)
        os.environ.pop("OTEL_SERVICE_NAME", None)

    def test_configures_logging_level(self):
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
        configure(trace_exporter="none")
        assert os.environ.get("SENTINAL_TRACING_DISABLED") == "1"

    def test_service_name_sets_env(self):
        os.environ.pop("OTEL_SERVICE_NAME", None)
        configure(service_name="my-agent")
        assert os.environ.get("OTEL_SERVICE_NAME") == "my-agent"

    def test_does_not_override_existing_service_name(self):
        os.environ["OTEL_SERVICE_NAME"] = "already-set"
        configure(service_name="new-name")
        assert os.environ["OTEL_SERVICE_NAME"] == "already-set"

    def test_console_exporter_removes_otlp_endpoint(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        configure(trace_exporter="console")
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ