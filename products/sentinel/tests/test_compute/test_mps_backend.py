from __future__ import annotations

import pytest
import torch

from sentinal.compute.mps_backend import MPSBackend


@pytest.fixture
def backend():
    """Fixture providing an MPSBackend instance."""
    backend = MPSBackend()
    if not backend.is_available():
        pytest.skip("MPS not available on this system")
    return backend


def test_name():
    """Test that name returns 'mps'."""
    backend = MPSBackend()
    assert backend.name == "mps"


def test_is_available():
    """Test that is_available returns True (we know MPS works on this machine)."""
    backend = MPSBackend()
    assert backend.is_available() is True


@pytest.mark.asyncio
async def test_dispatch_runs_on_gpu(backend):
    """Test that dispatch executes a matmul on MPS device and returns correct result shape."""
    def matmul_on_gpu():
        a = torch.randn(1024, 1024, device="mps")
        b = torch.randn(1024, 1024, device="mps")
        result = torch.matmul(a, b)
        return result.shape

    shape = await backend.dispatch(matmul_on_gpu)
    assert shape == torch.Size([1024, 1024])


@pytest.mark.asyncio
async def test_dispatch_faster_than_cpu(backend):
    """Test that 2048x2048 matmul on GPU is faster than CPU (with GPU warmup)."""
    def gpu_matmul():
        a = torch.randn(2048, 2048, device="mps")
        b = torch.randn(2048, 2048, device="mps")
        return torch.matmul(a, b)

    def cpu_matmul():
        a = torch.randn(2048, 2048, device="cpu")
        b = torch.randn(2048, 2048, device="cpu")
        return torch.matmul(a, b)

    # Warmup GPU
    for _ in range(3):
        await backend.dispatch(gpu_matmul)

    # Profile GPU
    gpu_time = backend.profile(gpu_matmul)

    # Profile CPU
    cpu_time = backend.profile(cpu_matmul)

    # GPU should be faster than CPU
    assert gpu_time < cpu_time, f"GPU time {gpu_time} should be < CPU time {cpu_time}"


def test_profile_includes_sync(backend):
    """Test that profile measures elapsed time correctly with synchronization."""
    def simple_gpu_op():
        a = torch.randn(512, 512, device="mps")
        b = torch.randn(512, 512, device="mps")
        return torch.matmul(a, b)

    elapsed = backend.profile(simple_gpu_op)
    assert elapsed > 0, f"Expected positive elapsed time, got {elapsed}"


@pytest.mark.asyncio
async def test_dispatch_propagates_exceptions(backend):
    """Test that RuntimeError propagates through dispatch."""
    def failing_fn():
        raise RuntimeError("Test error")

    with pytest.raises(RuntimeError, match="Test error"):
        await backend.dispatch(failing_fn)


def test_graceful_when_no_mps(monkeypatch):
    """Test that backend gracefully handles when MPS is not available."""
    # Monkeypatch _torch_available to False
    import sentinal.compute.mps_backend as mps_module
    monkeypatch.setattr(mps_module, "_torch_available", False)

    backend = MPSBackend()
    assert backend.is_available() is False
