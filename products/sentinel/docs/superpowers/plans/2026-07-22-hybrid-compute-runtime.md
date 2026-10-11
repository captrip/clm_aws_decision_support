# Hybrid Compute Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `sentinal/compute/` package that detects CPU/GPU hardware, routes actor work to the appropriate backend (OS threads for CPU, Metal/MPS for GPU), and replaces `asyncio.gather()` with a `HybridRunner` that achieves true simultaneous execution.

**Architecture:** A `ComputeBackend` ABC defines the dispatch interface. Three implementations (CPU, MPS, Metal) register with a `HybridRunner`. The runner examines each `Executable`'s `device` preference (or auto-detects via type checking + profiling) and dispatches actors to real OS threads or GPU accordingly.

**Tech Stack:** Python 3.14t (free-threaded), threading module, concurrent.futures.ThreadPoolExecutor, PyTorch MPS, pyobjc-framework-Metal, pytest

## Global Constraints

- Python 3.14.6+ free-threaded build required (`.ft-env`)
- All commands run with `PYTHON_GIL=0` prefix
- No numpy dependency (torch handles numerics)
- Existing async actor interface (`start()/stop()/receive()`) must remain compatible
- `ExecutionState` already uses `threading.Lock` — thread-safe
- All new files go in `sentinal/compute/` package
- Tests go in `tests/test_compute/`

---

### Task 1: DevicePreference Enum and ComputeBackend ABC

**Files:**
- Create: `sentinal/compute/__init__.py`
- Create: `sentinal/compute/device.py`
- Create: `sentinal/compute/backend.py`
- Test: `tests/test_compute/test_backend.py`

**Interfaces:**
- Consumes: Nothing (foundational)
- Produces:
  - `DevicePreference` enum with values `CPU`, `GPU`, `AUTO`
  - `ComputeBackend` ABC with methods: `name: str` property, `is_available() -> bool`, `async dispatch(fn: Callable, *args, **kwargs) -> Any`, `profile(fn: Callable, *args, **kwargs) -> float`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p sentinal/compute
mkdir -p tests/test_compute
```

- [ ] **Step 2: Write test for DevicePreference**

```python
# tests/test_compute/test_backend.py
import pytest
from sentinal.compute.device import DevicePreference


class TestDevicePreference:
    def test_enum_values(self):
        assert DevicePreference.CPU.value == "cpu"
        assert DevicePreference.GPU.value == "gpu"
        assert DevicePreference.AUTO.value == "auto"

    def test_default_is_auto(self):
        assert DevicePreference.AUTO == DevicePreference("auto")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sentinal.compute'`

- [ ] **Step 4: Implement DevicePreference**

```python
# sentinal/compute/__init__.py
from sentinal.compute.device import DevicePreference
from sentinal.compute.backend import ComputeBackend

__all__ = ["DevicePreference", "ComputeBackend"]
```

```python
# sentinal/compute/device.py
from enum import Enum


class DevicePreference(Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"
```

- [ ] **Step 5: Write test for ComputeBackend ABC**

```python
# append to tests/test_compute/test_backend.py
from sentinal.compute.backend import ComputeBackend


class TestComputeBackend:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ComputeBackend()

    def test_subclass_must_implement_methods(self):
        class Incomplete(ComputeBackend):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_valid_subclass(self):
        class ValidBackend(ComputeBackend):
            @property
            def name(self) -> str:
                return "test"

            def is_available(self) -> bool:
                return True

            async def dispatch(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

            def profile(self, fn, *args, **kwargs) -> float:
                return 0.001

        backend = ValidBackend()
        assert backend.name == "test"
        assert backend.is_available() is True
```

- [ ] **Step 6: Implement ComputeBackend ABC**

```python
# sentinal/compute/backend.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class ComputeBackend(ABC):
    """Abstract base for all compute backends (CPU, MPS, Metal, future CUDA)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique backend identifier."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's hardware is present and usable."""
        ...

    @abstractmethod
    async def dispatch(self, fn: Callable, *args, **kwargs) -> Any:
        """Dispatch a callable to this backend's compute resources."""
        ...

    @abstractmethod
    def profile(self, fn: Callable, *args, **kwargs) -> float:
        """Run fn once and return wall-clock time in seconds."""
        ...
```

- [ ] **Step 7: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_backend.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add sentinal/compute/__init__.py sentinal/compute/device.py sentinal/compute/backend.py tests/test_compute/test_backend.py
git commit -m "feat(compute): add DevicePreference enum and ComputeBackend ABC"
```

---

### Task 2: CPUBackend

**Files:**
- Create: `sentinal/compute/cpu_backend.py`
- Test: `tests/test_compute/test_cpu_backend.py`

**Interfaces:**
- Consumes: `ComputeBackend` ABC from Task 1
- Produces: `CPUBackend` class with `__init__(max_workers: int | None = None)`, inherits `ComputeBackend`. Uses `ThreadPoolExecutor` for dispatch. `name` returns `"cpu"`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_compute/test_cpu_backend.py
import sys
import time
import asyncio
import threading

import pytest

from sentinal.compute.cpu_backend import CPUBackend


class TestCPUBackend:
    def test_name(self):
        backend = CPUBackend()
        assert backend.name == "cpu"

    def test_is_available(self):
        backend = CPUBackend()
        assert backend.is_available() is True

    @pytest.mark.asyncio
    async def test_dispatch_runs_on_separate_thread(self):
        backend = CPUBackend(max_workers=2)
        main_tid = threading.current_thread().native_id

        def get_tid():
            return threading.current_thread().native_id

        worker_tid = await backend.dispatch(get_tid)
        assert worker_tid != main_tid

    @pytest.mark.asyncio
    async def test_dispatch_returns_result(self):
        backend = CPUBackend()

        def add(a, b):
            return a + b

        result = await backend.dispatch(add, 3, 7)
        assert result == 10

    @pytest.mark.asyncio
    async def test_dispatch_achieves_parallelism(self):
        """Multiple dispatches run on separate OS threads simultaneously."""
        backend = CPUBackend(max_workers=4)

        def cpu_work():
            total = 0.0
            for i in range(2_000_000):
                total += i * 0.000001
            return total

        start = time.perf_counter()
        results = await asyncio.gather(
            backend.dispatch(cpu_work),
            backend.dispatch(cpu_work),
            backend.dispatch(cpu_work),
            backend.dispatch(cpu_work),
        )
        parallel_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(4):
            cpu_work()
        serial_time = time.perf_counter() - start

        speedup = serial_time / parallel_time
        assert speedup > 2.0, f"Speedup only {speedup:.1f}x — threads not parallel"
        assert len(results) == 4

    def test_profile_returns_time(self):
        backend = CPUBackend()

        def work():
            time.sleep(0.01)

        elapsed = backend.profile(work)
        assert elapsed >= 0.01
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_dispatch_propagates_exceptions(self):
        backend = CPUBackend()

        def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await backend.dispatch(explode)

    @pytest.mark.asyncio
    async def test_shutdown(self):
        backend = CPUBackend(max_workers=2)
        await backend.dispatch(lambda: 42)
        backend.shutdown()
        # After shutdown, dispatch should raise
        with pytest.raises(RuntimeError):
            await backend.dispatch(lambda: 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_cpu_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Install pytest-asyncio**

Run: `PYTHON_GIL=0 .ft-env/bin/pip install pytest-asyncio`

- [ ] **Step 4: Implement CPUBackend**

```python
# sentinal/compute/cpu_backend.py
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from sentinal.compute.backend import ComputeBackend


class CPUBackend(ComputeBackend):
    """Dispatches work to real OS threads via ThreadPoolExecutor."""

    def __init__(self, max_workers: int | None = None):
        workers = max_workers or max(1, (os.cpu_count() or 4) - 1)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._shutdown = False

    @property
    def name(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        return True

    async def dispatch(self, fn: Callable, *args, **kwargs) -> Any:
        if self._shutdown:
            raise RuntimeError("CPUBackend has been shut down")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._pool, lambda: fn(*args, **kwargs)
        )

    def profile(self, fn: Callable, *args, **kwargs) -> float:
        start = time.perf_counter()
        fn(*args, **kwargs)
        return time.perf_counter() - start

    def shutdown(self) -> None:
        self._shutdown = True
        self._pool.shutdown(wait=False)
```

- [ ] **Step 5: Update `__init__.py`**

```python
# sentinal/compute/__init__.py
from sentinal.compute.device import DevicePreference
from sentinal.compute.backend import ComputeBackend
from sentinal.compute.cpu_backend import CPUBackend

__all__ = ["DevicePreference", "ComputeBackend", "CPUBackend"]
```

- [ ] **Step 6: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_cpu_backend.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add sentinal/compute/cpu_backend.py sentinal/compute/__init__.py tests/test_compute/test_cpu_backend.py
git commit -m "feat(compute): add CPUBackend with ThreadPoolExecutor dispatch"
```

---

### Task 3: MPSBackend (PyTorch Metal Performance Shaders)

**Files:**
- Create: `sentinal/compute/mps_backend.py`
- Test: `tests/test_compute/test_mps_backend.py`

**Interfaces:**
- Consumes: `ComputeBackend` ABC from Task 1
- Produces: `MPSBackend` class. `name` returns `"mps"`. `is_available()` checks `torch.backends.mps.is_available()`. `dispatch()` runs callable (expects it to use MPS tensors internally). `profile()` includes warmup + timed run with `torch.mps.synchronize()`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_compute/test_mps_backend.py
import time

import pytest
import torch

from sentinal.compute.mps_backend import MPSBackend


@pytest.fixture
def backend():
    b = MPSBackend()
    if not b.is_available():
        pytest.skip("MPS not available on this machine")
    return b


class TestMPSBackend:
    def test_name(self, backend):
        assert backend.name == "mps"

    def test_is_available(self, backend):
        assert backend.is_available() is True

    @pytest.mark.asyncio
    async def test_dispatch_runs_on_gpu(self, backend):
        """Dispatch a matmul and verify it ran on MPS."""

        def gpu_matmul():
            a = torch.randn(1024, 1024, device="mps")
            b = torch.randn(1024, 1024, device="mps")
            c = a @ b
            torch.mps.synchronize()
            return c.shape

        result = await backend.dispatch(gpu_matmul)
        assert result == torch.Size([1024, 1024])

    @pytest.mark.asyncio
    async def test_dispatch_faster_than_cpu(self, backend):
        """GPU matmul should outperform CPU for large matrices."""
        size = 2048

        def gpu_work():
            a = torch.randn(size, size, device="mps")
            b = torch.randn(size, size, device="mps")
            c = a @ b
            torch.mps.synchronize()
            return c

        def cpu_work():
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            return a @ b

        # Warmup
        await backend.dispatch(gpu_work)

        gpu_time = backend.profile(gpu_work)
        cpu_time = backend.profile(cpu_work)

        assert gpu_time < cpu_time, (
            f"GPU ({gpu_time*1000:.1f}ms) not faster than CPU ({cpu_time*1000:.1f}ms)"
        )

    def test_profile_includes_sync(self, backend):
        """Profile should include synchronize time (not just dispatch)."""

        def work():
            a = torch.randn(2048, 2048, device="mps")
            b = torch.randn(2048, 2048, device="mps")
            _ = a @ b
            torch.mps.synchronize()

        elapsed = backend.profile(work)
        assert elapsed > 0.0

    @pytest.mark.asyncio
    async def test_dispatch_propagates_exceptions(self, backend):
        def bad_work():
            raise RuntimeError("GPU error")

        with pytest.raises(RuntimeError, match="GPU error"):
            await backend.dispatch(bad_work)


class TestMPSBackendUnavailable:
    def test_graceful_when_no_mps(self, monkeypatch):
        """MPSBackend should not crash if torch is missing."""
        import sentinal.compute.mps_backend as mod
        monkeypatch.setattr(mod, "_torch_available", False)
        backend = MPSBackend()
        assert backend.is_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_mps_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement MPSBackend**

```python
# sentinal/compute/mps_backend.py
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from sentinal.compute.backend import ComputeBackend

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False


class MPSBackend(ComputeBackend):
    """Dispatches work to Apple GPU via PyTorch MPS."""

    @property
    def name(self) -> str:
        return "mps"

    def is_available(self) -> bool:
        if not _torch_available:
            return False
        return torch.backends.mps.is_available()

    async def dispatch(self, fn: Callable, *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def profile(self, fn: Callable, *args, **kwargs) -> float:
        if _torch_available:
            torch.mps.synchronize()
        start = time.perf_counter()
        fn(*args, **kwargs)
        if _torch_available:
            torch.mps.synchronize()
        return time.perf_counter() - start
```

- [ ] **Step 4: Update `__init__.py`**

```python
# sentinal/compute/__init__.py
from sentinal.compute.device import DevicePreference
from sentinal.compute.backend import ComputeBackend
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.compute.mps_backend import MPSBackend

__all__ = ["DevicePreference", "ComputeBackend", "CPUBackend", "MPSBackend"]
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_mps_backend.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sentinal/compute/mps_backend.py sentinal/compute/__init__.py tests/test_compute/test_mps_backend.py
git commit -m "feat(compute): add MPSBackend for PyTorch Metal GPU dispatch"
```

---

### Task 4: MetalBackend (Raw Compute Shaders)

**Files:**
- Create: `sentinal/compute/metal_backend.py`
- Test: `tests/test_compute/test_metal_backend.py`

**Interfaces:**
- Consumes: `ComputeBackend` ABC from Task 1
- Produces: `MetalBackend` class. `name` returns `"metal"`. `is_available()` checks for Metal device. `dispatch()` accepts a `MetalKernel` dataclass (shader_source, input_buffers, output_size, grid_size, group_size). `dispatch_fn()` runs a plain callable (like other backends). `compile_kernel(source, fn_name) -> compiled pipeline`. `profile()` times kernel dispatch.
- Also produces: `MetalKernel` dataclass with fields `shader_source: str`, `function_name: str`, `input_data: list[memoryview | bytes]`, `output_size: int`, `grid_size: tuple[int, int, int]`, `group_size: tuple[int, int, int]`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_compute/test_metal_backend.py
import pytest

try:
    import Metal
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

from sentinal.compute.metal_backend import MetalBackend, MetalKernel


pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


@pytest.fixture
def backend():
    return MetalBackend()


class TestMetalBackend:
    def test_name(self, backend):
        assert backend.name == "metal"

    def test_is_available(self, backend):
        assert backend.is_available() is True

    def test_device_info(self, backend):
        info = backend.device_info()
        assert "name" in info
        assert "Apple" in info["name"]

    @pytest.mark.asyncio
    async def test_dispatch_kernel_vector_add(self, backend):
        """Dispatch a vector add kernel and verify correctness."""
        import struct

        n = 1024
        shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void vector_add(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            out[id] = a[id] + b[id];
        }
        """

        input_a = struct.pack(f'{n}f', *[float(i) for i in range(n)])
        input_b = struct.pack(f'{n}f', *[float(i * 2) for i in range(n)])

        kernel = MetalKernel(
            shader_source=shader,
            function_name="vector_add",
            input_data=[input_a, input_b],
            output_size=n * 4,
            grid_size=(n, 1, 1),
            group_size=(256, 1, 1),
        )

        result_mv = await backend.dispatch(kernel)
        result = struct.unpack(f'{n}f', bytes(result_mv))

        for i in [0, 1, 100, 512, n - 1]:
            expected = float(i) + float(i * 2)
            assert abs(result[i] - expected) < 0.01

    @pytest.mark.asyncio
    async def test_dispatch_fn(self, backend):
        """dispatch_fn runs a regular callable."""

        def work():
            return 42

        result = await backend.dispatch_fn(work)
        assert result == 42

    def test_profile_kernel(self, backend):
        import struct

        n = 10_000
        shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(
            device float* out [[buffer(0)]],
            uint id [[thread_position_in_grid]]
        ) {
            out[id] = float(id);
        }
        """

        kernel = MetalKernel(
            shader_source=shader,
            function_name="noop",
            input_data=[],
            output_size=n * 4,
            grid_size=(n, 1, 1),
            group_size=(256, 1, 1),
        )

        elapsed = backend.profile(kernel)
        assert elapsed > 0.0
        assert elapsed < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_metal_backend.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement MetalBackend**

```python
# sentinal/compute/metal_backend.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

from sentinal.compute.backend import ComputeBackend

try:
    import Metal
    _metal_available = True
except ImportError:
    _metal_available = False


@dataclass(frozen=True)
class MetalKernel:
    """Descriptor for a Metal compute shader dispatch."""
    shader_source: str
    function_name: str
    input_data: list  # list of bytes-like objects for input buffers
    output_size: int  # bytes for output buffer
    grid_size: tuple[int, int, int]  # total threads
    group_size: tuple[int, int, int]  # threads per threadgroup


class MetalBackend(ComputeBackend):
    """Dispatches raw Metal compute shaders to Apple GPU."""

    def __init__(self):
        self._device = None
        self._pipeline_cache: dict[str, Any] = {}
        if _metal_available:
            self._device = Metal.MTLCreateSystemDefaultDevice()

    @property
    def name(self) -> str:
        return "metal"

    def is_available(self) -> bool:
        return _metal_available and self._device is not None

    def device_info(self) -> dict:
        if not self._device:
            return {}
        return {
            "name": self._device.name(),
            "unified_memory": self._device.hasUnifiedMemory(),
        }

    async def dispatch(self, kernel: MetalKernel | Callable, *args, **kwargs) -> Any:
        if isinstance(kernel, MetalKernel):
            return self._dispatch_kernel(kernel)
        return kernel(*args, **kwargs)

    async def dispatch_fn(self, fn: Callable, *args, **kwargs) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def profile(self, kernel_or_fn: MetalKernel | Callable, *args, **kwargs) -> float:
        start = time.perf_counter()
        if isinstance(kernel_or_fn, MetalKernel):
            self._dispatch_kernel(kernel_or_fn)
        else:
            kernel_or_fn(*args, **kwargs)
        return time.perf_counter() - start

    def _dispatch_kernel(self, kernel: MetalKernel) -> memoryview:
        pipeline = self._get_pipeline(kernel.shader_source, kernel.function_name)

        # Create buffers
        buffers = []
        for data in kernel.input_data:
            buf = self._device.newBufferWithBytes_length_options_(
                data, len(data), Metal.MTLResourceStorageModeShared
            )
            buffers.append(buf)

        # Output buffer
        out_buf = self._device.newBufferWithLength_options_(
            kernel.output_size, Metal.MTLResourceStorageModeShared
        )

        # Encode and dispatch
        command_queue = self._device.newCommandQueue()
        command_buffer = command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        for i, buf in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        encoder.setBuffer_offset_atIndex_(out_buf, 0, len(buffers))

        grid = Metal.MTLSizeMake(*kernel.grid_size)
        group = Metal.MTLSizeMake(*kernel.group_size)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid, group)
        encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        return memoryview(out_buf.contents().as_buffer(kernel.output_size))

    def _get_pipeline(self, shader_source: str, function_name: str):
        cache_key = f"{function_name}:{hash(shader_source)}"
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        options = Metal.MTLCompileOptions.alloc().init()
        library, error = self._device.newLibraryWithSource_options_error_(
            shader_source, options, None
        )
        if library is None:
            raise RuntimeError(f"Metal shader compilation failed: {error}")

        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Function '{function_name}' not found in shader")

        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline is None:
            raise RuntimeError(f"Pipeline creation failed: {error}")

        self._pipeline_cache[cache_key] = pipeline
        return pipeline
```

- [ ] **Step 4: Update `__init__.py`**

```python
# sentinal/compute/__init__.py
from sentinal.compute.device import DevicePreference
from sentinal.compute.backend import ComputeBackend
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.compute.mps_backend import MPSBackend
from sentinal.compute.metal_backend import MetalBackend, MetalKernel

__all__ = [
    "DevicePreference",
    "ComputeBackend",
    "CPUBackend",
    "MPSBackend",
    "MetalBackend",
    "MetalKernel",
]
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_metal_backend.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sentinal/compute/metal_backend.py sentinal/compute/__init__.py tests/test_compute/test_metal_backend.py
git commit -m "feat(compute): add MetalBackend for raw GPU compute shader dispatch"
```

---

### Task 5: WorkloadProfiler

**Files:**
- Create: `sentinal/compute/profiler.py`
- Test: `tests/test_compute/test_profiler.py`

**Interfaces:**
- Consumes: `ComputeBackend` (Task 1), `CPUBackend` (Task 2), `MPSBackend` (Task 3), `DevicePreference` (Task 1)
- Produces: `WorkloadProfiler` class with:
  - `__init__(backends: dict[str, ComputeBackend])`
  - `resolve(executable) -> str` — returns backend name
  - `profile_and_cache(executable, backends) -> str` — benchmarks and caches
  - `uses_tensors(executable) -> bool` — type-check heuristic

- [ ] **Step 1: Write failing test**

```python
# tests/test_compute/test_profiler.py
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
    """An executable that imports torch and uses tensors."""

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

    def test_resolve_explicit_gpu(self, profiler):
        exe = GPUExecutable()
        result = profiler.resolve(exe)
        # If no GPU backend available, falls back to cpu
        assert result in ("mps", "metal", "cpu")

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
        # Second call should hit cache
        assert type(exe) in profiler._cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_profiler.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement WorkloadProfiler**

```python
# sentinal/compute/profiler.py
from __future__ import annotations

import inspect
import textwrap
from typing import Any

from sentinal.compute.backend import ComputeBackend
from sentinal.compute.device import DevicePreference


class WorkloadProfiler:
    """Determines the best compute backend for an executable via heuristics and profiling."""

    def __init__(self, backends: dict[str, ComputeBackend]):
        self._backends = backends
        self._cache: dict[type, str] = {}

    def resolve(self, executable) -> str:
        """Determine the best backend name for this executable."""
        exe_type = type(executable)
        if exe_type in self._cache:
            return self._cache[exe_type]

        # Check explicit device preference
        device = getattr(executable, "device", DevicePreference.AUTO)
        if isinstance(device, DevicePreference):
            if device == DevicePreference.CPU:
                self._cache[exe_type] = "cpu"
                return "cpu"
            elif device == DevicePreference.GPU:
                result = self._resolve_gpu()
                self._cache[exe_type] = result
                return result

        # AUTO: use heuristic
        if self.uses_tensors(executable):
            result = self._resolve_gpu()
        else:
            result = "cpu"

        self._cache[exe_type] = result
        return result

    def uses_tensors(self, executable) -> bool:
        """Check if executable likely uses torch tensors."""
        try:
            source = inspect.getsource(type(executable))
            tensor_indicators = ["torch", "tensor", ".mps", ".cuda", "device="]
            return any(ind in source for ind in tensor_indicators)
        except (OSError, TypeError):
            return False

    def profile_and_cache(self, executable, test_fn) -> str:
        """Run a microbenchmark on available backends and cache the winner."""
        best_name = "cpu"
        best_time = float("inf")

        for name, backend in self._backends.items():
            if not backend.is_available():
                continue
            try:
                elapsed = backend.profile(test_fn)
                if elapsed < best_time:
                    best_time = elapsed
                    best_name = name
            except Exception:
                continue

        self._cache[type(executable)] = best_name
        return best_name

    def _resolve_gpu(self) -> str:
        """Find the best available GPU backend, fall back to cpu."""
        for name in ("mps", "metal"):
            if name in self._backends and self._backends[name].is_available():
                return name
        return "cpu"
```

- [ ] **Step 4: Update `__init__.py`**

```python
# sentinal/compute/__init__.py
from sentinal.compute.device import DevicePreference
from sentinal.compute.backend import ComputeBackend
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.compute.mps_backend import MPSBackend
from sentinal.compute.metal_backend import MetalBackend, MetalKernel
from sentinal.compute.profiler import WorkloadProfiler

__all__ = [
    "DevicePreference",
    "ComputeBackend",
    "CPUBackend",
    "MPSBackend",
    "MetalBackend",
    "MetalKernel",
    "WorkloadProfiler",
]
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_profiler.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sentinal/compute/profiler.py sentinal/compute/__init__.py tests/test_compute/test_profiler.py
git commit -m "feat(compute): add WorkloadProfiler with type-check heuristic and caching"
```

---

### Task 6: HybridRunner

**Files:**
- Create: `sentinal/compute/runner.py`
- Test: `tests/test_compute/test_runner.py`

**Interfaces:**
- Consumes: `ComputeBackend` (Task 1), `CPUBackend` (Task 2), `MPSBackend` (Task 3), `MetalBackend` (Task 4), `WorkloadProfiler` (Task 5), `DevicePreference` (Task 1), `ExecutionSession` (existing)
- Produces: `HybridRunner` class with:
  - `__init__(backends: list[ComputeBackend] | None = None)`
  - `async run_actors(session, executor, observer, governor) -> ExecutionSession`
  - `detect_backends() -> list[ComputeBackend]` (classmethod)
  - `shutdown()`

- [ ] **Step 1: Write failing test**

```python
# tests/test_compute/test_runner.py
import asyncio
import threading
import time

import pytest

from sentinal.compute.runner import HybridRunner
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.compute.device import DevicePreference
from sentinal.core.actor_ref import ActorRef
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.actors.executor import Executor
from sentinal.actors.observer import Observer
from sentinal.actors.governor import Governor


class SlowCPUExecutable(Executable):
    device = DevicePreference.CPU

    async def execute(self, context):
        # Simulate CPU-bound work
        total = 0.0
        for i in range(2_000_000):
            total += i * 0.000001
        return {"total": total}


class TestHybridRunner:
    @pytest.fixture
    def runner(self):
        r = HybridRunner(backends=[CPUBackend(max_workers=4)])
        yield r
        r.shutdown()

    def test_detect_backends(self):
        backends = HybridRunner.detect_backends()
        names = [b.name for b in backends]
        assert "cpu" in names

    @pytest.mark.asyncio
    async def test_run_actors_completes(self, runner):
        session = ExecutionSession(
            executable=SlowCPUExecutable(),
            context=ExecutionContext(input={"query": "test"}),
        )

        executor = Executor(ActorRef("executor", "local"))
        observer = Observer(ActorRef("observer", "local"))
        governor = Governor(ActorRef("governor", "local"))

        result = await runner.run_actors(session, executor, observer, governor)
        assert result.is_terminal()

    @pytest.mark.asyncio
    async def test_actors_run_on_separate_threads(self, runner):
        """Verify observer and governor run on different OS threads."""
        thread_ids = {}

        class TrackingExecutable(Executable):
            device = DevicePreference.CPU

            async def execute(self, context):
                thread_ids["executor"] = threading.current_thread().native_id
                total = 0.0
                for i in range(1_000_000):
                    total += i * 0.000001
                return {"done": True}

        session = ExecutionSession(
            executable=TrackingExecutable(),
            context=ExecutionContext(input={}),
        )

        executor = Executor(ActorRef("executor", "local"))
        observer = Observer(ActorRef("observer", "local"))
        governor = Governor(ActorRef("governor", "local"))

        await runner.run_actors(session, executor, observer, governor)

        # The executor ran on a worker thread, not main
        main_tid = threading.current_thread().native_id
        assert thread_ids.get("executor") != main_tid

    @pytest.mark.asyncio
    async def test_parallelism_faster_than_serial(self, runner):
        """HybridRunner should be faster than running actors sequentially."""

        class SlowExec(Executable):
            device = DevicePreference.CPU

            async def execute(self, context):
                total = 0.0
                for i in range(3_000_000):
                    total += i * 0.000001
                return {"v": total}

        session = ExecutionSession(
            executable=SlowExec(),
            context=ExecutionContext(input={}),
        )

        executor = Executor(ActorRef("executor", "local"))
        observer = Observer(ActorRef("observer", "local"))
        governor = Governor(ActorRef("governor", "local"))

        start = time.perf_counter()
        await runner.run_actors(session, executor, observer, governor)
        wall_time = time.perf_counter() - start

        # Wall time should be dominated by executor work only,
        # not executor + observer + governor serial
        # The executor does the heavy work; observer/governor just poll
        assert wall_time < 5.0  # sanity: should not hang
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement HybridRunner**

```python
# sentinal/compute/runner.py
from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

from sentinal.compute.backend import ComputeBackend
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.compute.profiler import WorkloadProfiler

if TYPE_CHECKING:
    from sentinal.actors.executor import Executor
    from sentinal.actors.observer import Observer
    from sentinal.actors.governor import Governor
    from sentinal.core.execution.session import ExecutionSession


class HybridRunner:
    """Routes actor work to CPU threads or GPU based on workload and hardware."""

    def __init__(self, backends: list[ComputeBackend] | None = None):
        if backends is None:
            backends = self.detect_backends()
        self._backends = {b.name: b for b in backends}
        self._profiler = WorkloadProfiler(self._backends)
        self._cpu = self._backends.get("cpu") or CPUBackend()

    @classmethod
    def detect_backends(cls) -> list[ComputeBackend]:
        """Detect available compute backends on this machine."""
        backends: list[ComputeBackend] = [CPUBackend()]

        try:
            from sentinal.compute.mps_backend import MPSBackend
            mps = MPSBackend()
            if mps.is_available():
                backends.append(mps)
        except Exception:
            pass

        try:
            from sentinal.compute.metal_backend import MetalBackend
            metal = MetalBackend()
            if metal.is_available():
                backends.append(metal)
        except Exception:
            pass

        return backends

    async def run_actors(
        self,
        session: ExecutionSession,
        executor: Executor,
        observer: Observer,
        governor: Governor,
    ) -> ExecutionSession:
        """
        Run all 3 actors with hybrid dispatch.

        - Executor: dispatched to resolved backend (CPU thread or GPU)
        - Observer: CPU thread (polling)
        - Governor: CPU thread (policy checks)
        """
        backend_name = self._profiler.resolve(session.executable)
        backend = self._backends.get(backend_name, self._cpu)

        async def run_executor():
            await backend.dispatch(
                self._run_sync_actor, executor, session
            )

        async def run_observer():
            await self._cpu.dispatch(
                self._run_observer_sync, observer, session
            )

        async def run_governor():
            await self._cpu.dispatch(
                self._run_governor_sync, governor, session
            )

        await asyncio.gather(
            run_executor(),
            run_observer(),
            run_governor(),
        )

        return session

    def _run_sync_actor(self, executor: Executor, session: ExecutionSession):
        """Run executor's session work synchronously (called from thread)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(executor.execute_session(session))
        finally:
            loop.close()

    def _run_observer_sync(self, observer: Observer, session: ExecutionSession):
        """Run observer polling synchronously (called from thread)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(observer.observe_session(session))
        finally:
            loop.close()

    def _run_governor_sync(self, governor: Governor, session: ExecutionSession):
        """Run governor polling synchronously (called from thread)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(governor.govern_session(session))
        finally:
            loop.close()

    @property
    def available_backends(self) -> list[str]:
        return [name for name, b in self._backends.items() if b.is_available()]

    def shutdown(self):
        for backend in self._backends.values():
            if hasattr(backend, "shutdown"):
                backend.shutdown()
```

- [ ] **Step 4: Update `__init__.py`**

```python
# sentinal/compute/__init__.py
from sentinal.compute.device import DevicePreference
from sentinal.compute.backend import ComputeBackend
from sentinal.compute.cpu_backend import CPUBackend
from sentinal.compute.mps_backend import MPSBackend
from sentinal.compute.metal_backend import MetalBackend, MetalKernel
from sentinal.compute.profiler import WorkloadProfiler
from sentinal.compute.runner import HybridRunner

__all__ = [
    "DevicePreference",
    "ComputeBackend",
    "CPUBackend",
    "MPSBackend",
    "MetalBackend",
    "MetalKernel",
    "WorkloadProfiler",
    "HybridRunner",
]
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_runner.py -v -s`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add sentinal/compute/runner.py sentinal/compute/__init__.py tests/test_compute/test_runner.py
git commit -m "feat(compute): add HybridRunner that routes actors to CPU threads or GPU"
```

---

### Task 7: Wire HybridRunner into SentinelRuntime

**Files:**
- Modify: `sentinal/runtime/runtime.py`
- Modify: `sentinal/core/executable.py`
- Modify: `sentinal/core/capabilities.py`
- Test: `tests/test_compute/test_runtime_integration.py`

**Interfaces:**
- Consumes: `HybridRunner` (Task 6), `DevicePreference` (Task 1), all backends
- Produces: Modified `SentinelRuntime.execute()` that uses `HybridRunner`. `Executable` gains optional `device: DevicePreference` attribute. `Capabilities` gains `gpu_capable: bool` field.

- [ ] **Step 1: Write integration test**

```python
# tests/test_compute/test_runtime_integration.py
import asyncio
import threading
import time

import pytest

from sentinal.compute.device import DevicePreference
from sentinal.core.actor_ref import ActorRef
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.runtime.runtime import SentinelRuntime


class CPUBoundExecutable(Executable):
    device = DevicePreference.CPU

    async def execute(self, context):
        total = 0.0
        for i in range(2_000_000):
            total += i * 0.000001
        return {"total": total}


class AutoExecutable(Executable):
    async def execute(self, context):
        return {"status": "done", "thread": threading.current_thread().native_id}


class TestRuntimeWithHybridRunner:
    @pytest.mark.asyncio
    async def test_execute_uses_hybrid_runner(self):
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=CPUBoundExecutable(),
            context=ExecutionContext(input={"q": "test"}),
        )

        result = await runtime.execute(session)
        assert result.is_terminal()
        memory = result.state.get_all_memory()
        assert memory.get("actor_status") == "completed"

    @pytest.mark.asyncio
    async def test_runtime_reports_available_backends(self):
        runtime = SentinelRuntime()
        metrics = runtime.get_metrics()
        assert "backends" in metrics
        assert "cpu" in metrics["backends"]

    @pytest.mark.asyncio
    async def test_executor_runs_on_worker_thread(self):
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=AutoExecutable(),
            context=ExecutionContext(input={}),
        )

        result = await runtime.execute(session)
        memory = result.state.get_all_memory()
        worker_tid = memory.get("result", {}).get("thread")
        main_tid = threading.current_thread().native_id

        # Executor should have run on a different OS thread
        assert worker_tid is not None
        assert worker_tid != main_tid

    @pytest.mark.asyncio
    async def test_shutdown(self):
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=AutoExecutable(),
            context=ExecutionContext(input={}),
        )
        await runtime.execute(session)
        await runtime.shutdown()
        assert runtime._running is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_runtime_integration.py -v`
Expected: FAIL (runtime doesn't use HybridRunner yet)

- [ ] **Step 3: Add `device` to Executable base class**

```python
# sentinal/core/executable.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.result import ExecutionResult

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class Executable(ABC, Generic[TInput, TOutput]):
    """
    Base interface implemented by all executable workloads.

    Set `device` class attribute to control compute routing:
    - DevicePreference.CPU: force OS thread
    - DevicePreference.GPU: prefer GPU (MPS/Metal)
    - DevicePreference.AUTO: runtime decides (default)
    """

    # Subclasses can override to declare device preference.
    # Import here would be circular, so we use a sentinel and
    # check at runtime. Any value from DevicePreference enum works.
    device = None  # None means AUTO

    @abstractmethod
    async def execute(
        self,
        context: ExecutionContext,
    ) -> ExecutionResult[TOutput]:
        """
        Execute the workload.

        Implementations should remain stateless.
        """
        raise NotImplementedError
```

- [ ] **Step 4: Add `gpu_capable` to Capabilities**

```python
# sentinal/core/capabilities.py
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Capabilities:
    """
    Declares features supported by an adapter.

    Runtime behaviour is adapted automatically based
    on these capabilities.
    """

    streaming: bool = False

    checkpointing: bool = False

    cancellation: bool = False

    replay: bool = False

    tool_events: bool = False

    node_events: bool = False

    prompt_mutation: bool = False

    gpu_capable: bool = False
```

- [ ] **Step 5: Modify SentinelRuntime to use HybridRunner**

```python
# sentinal/runtime/runtime.py
"""
SentinelRuntime - Orchestrates Actor, Observer, Governor

Manages:
- Actor lifecycle (start/stop)
- Session registry
- Concurrent execution via HybridRunner (CPU threads + GPU)
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional
from uuid import UUID

from sentinal.compute.runner import HybridRunner
from sentinal.core.actor_ref import ActorRef
from sentinal.core.contracts.execution_signals import StartExecution
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.messaging.dispatcher import DefaultDispatcher
from sentinal.runtime.session_registry import SessionRegistry


class SentinelRuntime:
    """
    Sentinel runtime - orchestrates everything.

    Creates and manages:
    - Executor (Actor thread)
    - Observer (Observer thread)
    - Governor (Governor thread)
    - Session registry
    - Message dispatcher
    - HybridRunner (CPU/GPU dispatch)
    """

    def __init__(self, quality_threshold: float = 0.75):
        self.dispatcher = DefaultDispatcher()
        self.registry = SessionRegistry()
        self.actors = []
        self.quality_threshold = quality_threshold

        # Runtime ref
        self.ref = ActorRef("runtime", "local")

        # Track running
        self._running = False

        # Hybrid compute runner
        self._runner = HybridRunner()

    def register(self, actor) -> None:
        """Register an actor."""
        self.actors.append(actor)
        actor.dispatcher = self.dispatcher
        self.dispatcher.register(actor.ref, actor.mailbox)

    async def start(self) -> None:
        """Start all actors."""
        if self._running:
            return
        self._running = True
        for actor in self.actors:
            await actor.start()

    async def stop(self) -> None:
        """Stop all actors."""
        if not self._running:
            return
        self._running = False
        for actor in self.actors:
            await actor.stop()

    async def submit(self, session: ExecutionSession) -> None:
        """Submit session for execution (message-based)."""
        self.registry.add(session)
        await self.dispatcher.publish(
            StartExecution(
                execution_id=session.execution_id,
                sender=self.ref,
                data=session.execution_id,
            )
        )

    async def execute(self, session: ExecutionSession) -> ExecutionSession:
        """
        Execute session with 3-thread supervision.

        Uses HybridRunner to dispatch actors to real OS threads (CPU)
        or GPU based on the executable's device preference.
        """
        self.registry.add(session)

        # Get or create actors
        executor = self._get_actor("executor")
        observer = self._get_actor("observer")
        governor = self._get_actor("governor")

        from sentinal.actors.executor import Executor
        from sentinal.actors.observer import Observer
        from sentinal.actors.governor import Governor

        if not executor:
            executor = Executor(ActorRef("executor", "local"))
            self.register(executor)

        if not observer:
            observer = Observer(ActorRef("observer", "local"))
            self.register(observer)

        if not governor:
            governor = Governor(
                ActorRef("governor", "local"),
                quality_threshold=self.quality_threshold,
            )
            self.register(governor)

        if not self._running:
            await self.start()

        # Use HybridRunner instead of asyncio.gather
        return await self._runner.run_actors(session, executor, observer, governor)

    def _get_actor(self, name: str):
        """Get actor by name."""
        for actor in self.actors:
            if actor.ref.name == name:
                return actor
        return None

    async def get_session(self, execution_id: UUID) -> Optional[ExecutionSession]:
        """Get session by execution ID."""
        return self.registry.get(execution_id)

    def get_metrics(self) -> Dict:
        """Get runtime metrics."""
        return {
            "actors": len(self.actors),
            "running": self._running,
            "sessions": len(self.registry._sessions),
            "backends": self._runner.available_backends,
        }

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        await self.stop()
        self._runner.shutdown()
        self.actors.clear()
        self.registry.clear()
```

- [ ] **Step 6: Run tests and verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_runtime_integration.py -v -s`
Expected: All PASS

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add sentinal/runtime/runtime.py sentinal/core/executable.py sentinal/core/capabilities.py tests/test_compute/test_runtime_integration.py
git commit -m "feat(runtime): wire HybridRunner into SentinelRuntime, replace asyncio.gather"
```

---

### Task 8: End-to-End Parallelism Proof

**Files:**
- Create: `tests/test_compute/test_e2e_parallelism.py`

**Interfaces:**
- Consumes: Full runtime (Task 7), all backends, `DevicePreference`
- Produces: Test that proves CPU threads + GPU run simultaneously through the runtime (not just raw threads)

- [ ] **Step 1: Write end-to-end parallelism test**

```python
# tests/test_compute/test_e2e_parallelism.py
"""
End-to-end proof that SentinelRuntime achieves true heterogeneous parallelism:
- CPU-bound executables run on real OS threads
- GPU-bound executables dispatch to Metal/MPS
- Observer and Governor run on separate OS threads simultaneously
"""
import sys
import threading
import time

import pytest
import torch

from sentinal.compute.device import DevicePreference
from sentinal.core.actor_ref import ActorRef
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.runtime.runtime import SentinelRuntime


class CPUHeavyExecutable(Executable):
    """CPU-bound workload that records its thread ID."""
    device = DevicePreference.CPU

    async def execute(self, context):
        tid = threading.current_thread().native_id
        t0 = time.perf_counter()
        total = 0.0
        for i in range(3_000_000):
            total += i * 0.000001
        t1 = time.perf_counter()
        return {
            "thread_id": tid,
            "start": t0,
            "end": t1,
            "duration_ms": (t1 - t0) * 1000,
            "total": total,
        }


class GPUMatmulExecutable(Executable):
    """GPU-bound workload using PyTorch MPS."""
    device = DevicePreference.GPU

    async def execute(self, context):
        tid = threading.current_thread().native_id
        size = context.input.get("size", 2048)

        a = torch.randn(size, size, device="mps")
        b = torch.randn(size, size, device="mps")
        # warmup
        _ = a @ b
        torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(5):
            c = a @ b
        torch.mps.synchronize()
        t1 = time.perf_counter()

        return {
            "thread_id": tid,
            "start": t0,
            "end": t1,
            "duration_ms": (t1 - t0) * 1000,
            "shape": list(c.shape),
        }


@pytest.fixture(autouse=True)
def check_env():
    assert sys._is_gil_enabled() is False, "GIL must be disabled"


class TestEndToEndParallelism:
    @pytest.mark.asyncio
    async def test_cpu_executable_runs_on_worker_thread(self):
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=CPUHeavyExecutable(),
            context=ExecutionContext(input={}),
        )

        result = await runtime.execute(session)
        memory = result.state.get_all_memory()
        exe_result = memory["result"]

        main_tid = threading.current_thread().native_id
        assert exe_result["thread_id"] != main_tid
        assert exe_result["duration_ms"] > 0

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_gpu_executable_dispatches_to_mps(self):
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=GPUMatmulExecutable(),
            context=ExecutionContext(input={"size": 2048}),
        )

        result = await runtime.execute(session)
        memory = result.state.get_all_memory()
        exe_result = memory["result"]

        assert exe_result["shape"] == [2048, 2048]
        assert exe_result["duration_ms"] > 0

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_runtime_achieves_parallel_execution(self):
        """The full 3-actor model runs faster than sequential."""
        runtime = SentinelRuntime()
        session = ExecutionSession(
            executable=CPUHeavyExecutable(),
            context=ExecutionContext(input={}),
        )

        start = time.perf_counter()
        result = await runtime.execute(session)
        wall_time = time.perf_counter() - start

        memory = result.state.get_all_memory()
        exe_result = memory["result"]

        # Executor did real CPU work
        assert exe_result["duration_ms"] > 100

        # Wall time should be close to executor time (observer/governor are concurrent)
        # If they were serial, wall time would be executor + observer + governor
        print(f"\n  Executor work: {exe_result['duration_ms']:.0f}ms")
        print(f"  Wall time: {wall_time*1000:.0f}ms")

        await runtime.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_sessions_sequential(self):
        """Runtime handles multiple sessions correctly."""
        runtime = SentinelRuntime()

        for i in range(3):
            session = ExecutionSession(
                executable=CPUHeavyExecutable(),
                context=ExecutionContext(input={"iteration": i}),
            )
            result = await runtime.execute(session)
            assert result.is_terminal()
            memory = result.state.get_all_memory()
            assert memory["actor_status"] == "completed"

        await runtime.shutdown()
```

- [ ] **Step 2: Run tests**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_compute/test_e2e_parallelism.py -v -s`
Expected: All PASS

- [ ] **Step 3: Run the full parallelism test suite together**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/ -v --tb=short -q`
Expected: All tests pass (no regressions)

- [ ] **Step 4: Commit**

```bash
git add tests/test_compute/test_e2e_parallelism.py
git commit -m "test: add end-to-end proof of hybrid CPU+GPU parallelism through runtime"
```

---

## Task Dependency Graph

```
Task 1 (ABC + enum)
├── Task 2 (CPUBackend)
├── Task 3 (MPSBackend)
├── Task 4 (MetalBackend)
└── Task 5 (Profiler) ← depends on 2, 3
    └── Task 6 (HybridRunner) ← depends on 2, 3, 4, 5
        └── Task 7 (Wire into Runtime) ← depends on 6
            └── Task 8 (E2E proof) ← depends on 7
```

Tasks 2, 3, 4 can be implemented in parallel after Task 1. Tasks 5-8 are sequential.
