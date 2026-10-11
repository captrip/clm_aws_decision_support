"""Tests for @traced and @logged decorators."""
from __future__ import annotations

import asyncio
import logging

import pytest

from sentinal.telemetry.decorators import traced, logged


class TestTracedDecorator:
    """Test the @traced decorator."""

    def test_sync_function(self):
        """Test @traced on a synchronous function."""
        @traced
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_async_function(self):
        """Test @traced on an async function."""
        @traced
        async def async_add(a, b):
            return a + b

        result = asyncio.run(async_add(2, 3))
        assert result == 5

    def test_with_custom_name(self):
        """Test @traced with custom span name."""
        @traced(name="custom.operation")
        def my_func():
            return "ok"

        result = my_func()
        assert result == "ok"

    def test_with_attributes(self):
        """Test @traced with custom attributes."""
        @traced(attributes={"env": "test", "version": "1.0"})
        def my_func():
            return "ok"

        result = my_func()
        assert result == "ok"

    def test_preserves_function_metadata(self):
        """Test that @traced preserves function name and docstring."""
        @traced
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_records_exception(self):
        """Test that @traced records exceptions."""
        @traced
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()

    def test_async_exception(self):
        """Test that @traced records exceptions in async functions."""
        @traced
        async def async_failing_func():
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            asyncio.run(async_failing_func())


class TestLoggedDecorator:
    """Test the @logged decorator."""

    def test_sync_function(self, caplog):
        """Test @logged on a synchronous function."""
        @logged
        def add(a, b):
            return a + b

        with caplog.at_level(logging.INFO):
            result = add(2, 3)
            assert result == 5

        # Check that entry and exit were logged
        messages = [record.message for record in caplog.records]
        assert any("Calling" in msg and "add" in msg for msg in messages)
        assert any("completed in" in msg and "ms" in msg and "add" in msg for msg in messages)

    def test_async_function(self, caplog):
        """Test @logged on an async function."""
        @logged
        async def async_add(a, b):
            return a + b

        with caplog.at_level(logging.INFO):
            result = asyncio.run(async_add(2, 3))
            assert result == 5

        messages = [record.message for record in caplog.records]
        assert any("Calling" in msg and "async_add" in msg for msg in messages)
        assert any("completed in" in msg and "async_add" in msg for msg in messages)

    def test_custom_level(self, caplog):
        """Test @logged with custom log level."""
        @logged(level="DEBUG")
        def my_func():
            return "ok"

        with caplog.at_level(logging.DEBUG):
            result = my_func()
            assert result == "ok"

        # Should see DEBUG level logs
        assert any(record.levelname == "DEBUG" for record in caplog.records)

    def test_capture_result(self, caplog):
        """Test @logged with capture_result=True."""
        @logged(capture_result=True)
        def compute():
            return 42

        with caplog.at_level(logging.INFO):
            result = compute()
            assert result == 42

        messages = [record.message for record in caplog.records]
        assert any("result=42" in msg for msg in messages)

    def test_logs_exception(self, caplog):
        """Test that @logged logs exceptions."""
        @logged
        def failing_func():
            raise ValueError("test error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                failing_func()

        messages = [record.message for record in caplog.records]
        assert any("failed after" in msg and "failing_func" in msg and "test error" in msg for msg in messages)

    def test_preserves_function_metadata(self):
        """Test that @logged preserves function name and docstring."""
        @logged
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."


class TestComposition:
    """Test composing @traced and @logged together."""

    def test_traced_and_logged_together(self, caplog):
        """Test that @traced and @logged can be used together on sync function."""
        @traced
        @logged
        def compute(x):
            return x * 2

        with caplog.at_level(logging.INFO):
            result = compute(5)
            assert result == 10

        # Verify logging happened
        messages = [record.message for record in caplog.records]
        assert any("Calling" in msg and "compute" in msg for msg in messages)
        assert any("completed in" in msg and "compute" in msg for msg in messages)

    def test_async_composition(self, caplog):
        """Test that @traced and @logged can be used together on async function."""
        @traced
        @logged
        async def async_compute(x):
            return x * 2

        with caplog.at_level(logging.INFO):
            result = asyncio.run(async_compute(5))
            assert result == 10

        # Verify logging happened
        messages = [record.message for record in caplog.records]
        assert any("Calling" in msg and "async_compute" in msg for msg in messages)
        assert any("completed in" in msg and "async_compute" in msg for msg in messages)