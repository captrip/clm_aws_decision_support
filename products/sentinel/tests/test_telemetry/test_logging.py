import json
import logging

import pytest

from sentinal.telemetry.logging import setup_logging, JSONFormatter


class TestSetupLogging:
    def teardown_method(self):
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
        logger = logging.getLogger("sentinal.test_text")
        logger.info("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.err

    def test_json_format_produces_valid_json(self, capsys):
        setup_logging(level="INFO", format="json")
        logger = logging.getLogger("sentinal.test_json")
        logger.info("test message")
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

    def test_no_duplicate_handlers(self):
        setup_logging()
        setup_logging()
        logger = logging.getLogger("sentinal")
        assert len(logger.handlers) == 1


class TestJSONFormatter:
    def test_format_includes_required_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="sentinal.test", level=logging.INFO, pathname="",
            lineno=0, msg="test msg", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "test msg"
        assert data["logger"] == "sentinal.test"
        assert data["level"] == "INFO"
        assert "timestamp" in data

    def test_includes_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="sentinal.test", level=logging.INFO, pathname="",
            lineno=0, msg="with extra", args=(), exc_info=None,
        )
        record.backend = "cpu"
        record.duration_ms = 42.5
        output = formatter.format(record)
        data = json.loads(output)
        assert data["backend"] == "cpu"
        assert data["duration_ms"] == 42.5
