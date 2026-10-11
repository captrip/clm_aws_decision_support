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
        # Include any extra fields passed via `extra={}`
        standard_keys = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in standard_keys and key not in ("message", "args"):
                data[key] = value
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
