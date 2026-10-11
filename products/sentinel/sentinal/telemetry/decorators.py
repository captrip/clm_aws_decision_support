"""User-facing decorators: @traced and @logged."""
from __future__ import annotations

import functools
import inspect
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

        if inspect.iscoroutinefunction(func):
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

        if inspect.iscoroutinefunction(func):
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
                    logger.error(f"{func.__qualname__} failed after {duration_ms:.1f}ms: {e}")
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
                    logger.error(f"{func.__qualname__} failed after {duration_ms:.1f}ms: {e}")
                    raise
            return sync_wrapper

    if fn is not None:
        return decorator(fn)
    return decorator