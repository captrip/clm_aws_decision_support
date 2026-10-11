import pytest
from sentinal.compute.device import DevicePreference
from sentinal.compute.profiler import WorkloadProfiler
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext


class CPUExecutable(Executable):
    device = DevicePreference.CPU
    async def execute(self, context):
        return {"result": "cpu"}


class GPUExecutable(Executable):
    device = DevicePreference.GPU
    async def execute(self, context):
        return {"result": "gpu"}


class AutoExecutable(Executable):
    async def execute(self, context):
        return {"result": "auto"}


class TensorExecutable(Executable):
    async def execute(self, context):
        import torch
        a = torch.randn(100, 100, device="mps")
        return {"result": a.shape}


class TestWorkloadProfiler:
    @pytest.fixture
    def profiler(self):
        backends = {"cpu": CPUBackend(max_workers=2)}
        return WorkloadProfiler(backends)

    def test_resolve_explicit_cpu(self, profiler):
        exe = CPUExecutable()
        assert profiler.resolve(exe) == "cpu"

    def test_resolve_explicit_gpu_falls_back(self, profiler):
        """With only CPU backend available, GPU preference falls back to cpu."""
        exe = GPUExecutable()
        assert profiler.resolve(exe) == "cpu"

    def test_resolve_auto_no_tensors(self, profiler):
        exe = AutoExecutable()
        assert profiler.resolve(exe) == "cpu"

    def test_uses_tensors_detection(self, profiler):
        assert profiler.uses_tensors(TensorExecutable()) is True
        assert profiler.uses_tensors(AutoExecutable()) is False

    def test_caches_result(self, profiler):
        exe = AutoExecutable()
        result1 = profiler.resolve(exe)
        result2 = profiler.resolve(exe)
        assert result1 == result2
        assert type(exe) in profiler._cache

    def test_resolve_gpu_with_mps_backend(self):
        """With MPS backend available, GPU preference resolves to mps."""
        from sentinal.compute.mps_backend import MPSBackend
        mps = MPSBackend()
        if not mps.is_available():
            pytest.skip("MPS not available")
        backends = {"cpu": CPUBackend(max_workers=2), "mps": mps}
        profiler = WorkloadProfiler(backends)
        exe = GPUExecutable()
        assert profiler.resolve(exe) == "mps"

    def test_profile_and_cache(self, profiler):
        import time
        exe = AutoExecutable()
        def test_fn():
            time.sleep(0.001)
        result = profiler.profile_and_cache(exe, test_fn)
        assert result == "cpu"
        assert type(exe) in profiler._cache
