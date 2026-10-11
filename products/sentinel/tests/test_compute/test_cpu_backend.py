from __future__ import annotations

import asyncio
import threading
import time

import pytest

from sentinal.compute.cpu_backend import CPUBackend


def test_name():
    """Test that CPUBackend name is 'cpu'."""
    backend = CPUBackend()
    assert backend.name == "cpu"
    backend.shutdown()


def test_is_available():
    """Test that CPUBackend is always available."""
    backend = CPUBackend()
    assert backend.is_available() is True
    backend.shutdown()


@pytest.mark.asyncio
async def test_dispatch_runs_on_separate_thread():
    """Test that dispatch runs on a different thread than the main thread."""
    backend = CPUBackend()
    main_thread_id = threading.current_thread().native_id

    def get_thread_id():
        return threading.current_thread().native_id

    worker_thread_id = await backend.dispatch(get_thread_id)
    assert worker_thread_id != main_thread_id, "Work should run on a separate thread"
    backend.shutdown()


@pytest.mark.asyncio
async def test_dispatch_returns_result():
    """Test that dispatch returns the correct result."""
    backend = CPUBackend()

    def add(a, b):
        return a + b

    result = await backend.dispatch(add, 3, 7)
    assert result == 10
    backend.shutdown()


@pytest.mark.asyncio
async def test_dispatch_achieves_parallelism():
    """Test that parallel dispatches achieve speedup over serial execution."""
    backend = CPUBackend(max_workers=4)

    def cpu_work():
        """Simulate CPU-intensive work."""
        total = 0
        for i in range(2_000_000):
            total += i
        return total

    # Serial execution
    serial_start = time.perf_counter()
    for _ in range(4):
        cpu_work()
    serial_time = time.perf_counter() - serial_start

    # Parallel execution
    parallel_start = time.perf_counter()
    tasks = [backend.dispatch(cpu_work) for _ in range(4)]
    await asyncio.gather(*tasks)
    parallel_time = time.perf_counter() - parallel_start

    speedup = serial_time / parallel_time
    assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"
    backend.shutdown()


def test_profile_returns_time():
    """Test that profile returns elapsed time >= expected duration."""
    backend = CPUBackend()

    def sleep_work():
        time.sleep(0.01)

    elapsed = backend.profile(sleep_work)
    assert elapsed >= 0.01, f"Expected >= 0.01s, got {elapsed:.4f}s"
    backend.shutdown()


@pytest.mark.asyncio
async def test_dispatch_propagates_exceptions():
    """Test that exceptions raised in dispatched work are propagated."""
    backend = CPUBackend()

    def raise_error():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        await backend.dispatch(raise_error)

    backend.shutdown()


@pytest.mark.asyncio
async def test_shutdown():
    """Test that dispatch raises RuntimeError after shutdown."""
    backend = CPUBackend()

    def dummy_work():
        return 42

    # Should work before shutdown
    result = await backend.dispatch(dummy_work)
    assert result == 42

    # Shutdown the backend
    backend.shutdown()

    # Should raise RuntimeError after shutdown
    with pytest.raises(RuntimeError, match="CPUBackend has been shut down"):
        await backend.dispatch(dummy_work)
