# Hybrid Compute Runtime — Design Spec

## Summary

Modify the Sentinel framework to dispatch actor work across CPU threads and GPU based on hardware availability and workload type. Leverages Python 3.14t (free-threaded, GIL disabled) for true OS-thread parallelism on CPU, and Metal/MPS for GPU acceleration — running simultaneously.

## Goals

1. Actors run on real OS threads (not just asyncio tasks) for CPU-bound work
2. GPU-capable workloads dispatch to Metal GPU automatically or by declaration
3. Runtime detects available hardware at startup and routes accordingly
4. Executables can declare device preference; undeclared ones get auto-routed
5. First-run profiling validates routing decisions and self-corrects
6. GPU backend is extensible (MPS today, CUDA/ROCm later without runtime changes)

## Non-Goals

- Multi-GPU support (single device for now)
- Distributed compute across machines
- Dynamic mid-execution device migration

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  SentinelRuntime                      │
├─────────────────────────────────────────────────────┤
│  HybridRunner                                        │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ OS Thread 1 │  │  OS Thread 2 │  │ Async GPU │  │
│  │  Observer   │  │  Governor    │  │  Executor │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  │
│         │                │               │          │
│         └────────────────┼───────────────┘          │
│                          ▼                           │
│              ComputeBackend (registry)               │
│         ┌────────────┬────────────────┐             │
│         ▼            ▼                ▼             │
│    CPUBackend    MPSBackend     MetalBackend        │
│   (ThreadPool)  (torch.mps)   (raw shaders)        │
└─────────────────────────────────────────────────────┘
```

## Components

### 1. DevicePreference (enum)

```python
class DevicePreference(Enum):
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"
```

Added as an optional class attribute on `Executable`. Defaults to `AUTO`.

### 2. ComputeBackend (abstract base)

```python
class ComputeBackend(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    async def dispatch(self, fn, *args, **kwargs) -> Any: ...

    @abstractmethod
    def profile(self, fn, *args, **kwargs) -> float:
        """Run a microbenchmark, return time in seconds."""
        ...
```

### 3. CPUBackend

- Wraps `concurrent.futures.ThreadPoolExecutor`
- Worker count = `os.cpu_count() - 1` (leave one for the event loop)
- `dispatch()` runs the callable on a real OS thread via `loop.run_in_executor`
- `is_available()` always True
- `profile()` runs the fn once on a thread, measures wall time

### 4. MPSBackend

- `is_available()` checks `torch.backends.mps.is_available()`
- `dispatch()` moves tensors to MPS device, runs the op, calls `torch.mps.synchronize()`
- `profile()` runs warmup + timed iteration on MPS
- Handles the case where torch is not installed (graceful fallback)

### 5. MetalBackend

- `is_available()` checks `Metal.MTLCreateSystemDefaultDevice() is not None`
- `dispatch()` accepts a `MetalKernel` descriptor (shader source + buffers + grid)
- Compiles shader, dispatches command buffer, waits for completion
- Returns output buffer as memoryview
- `profile()` dispatches kernel once, measures commit-to-completion time

### 6. HybridRunner

Replaces `asyncio.gather()` in `SentinelRuntime.execute()`.

```python
class HybridRunner:
    def __init__(self, backends: list[ComputeBackend]):
        self.backends = {b.name: b for b in backends}
        self.profiler = WorkloadProfiler()

    async def run_actors(self, session, executor, observer, governor):
        """Run all 3 actors with appropriate dispatch."""
        # Determine device for executor's workload
        device = self._resolve_device(session.executable)

        # Launch observer + governor on CPU threads
        # Launch executor on resolved device
        ...
```

Routing logic:
1. Check `executable.device` preference
2. If `AUTO`: type-check for tensor usage → GPU, else → CPU
3. On first run with AUTO: profiler benchmarks both paths, caches the winner

### 7. WorkloadProfiler

```python
class WorkloadProfiler:
    def __init__(self):
        self._cache: dict[type, DevicePreference] = {}

    def resolve(self, executable, backends) -> str:
        """Determine best backend for this executable type."""
        if type(executable) in self._cache:
            return self._cache[type(executable)]

        # Type check heuristic
        if self._uses_tensors(executable):
            choice = "mps"
        else:
            choice = "cpu"

        # Validate with microbenchmark on first run
        # (deferred to background after first execution)
        return choice

    def _uses_tensors(self, executable) -> bool:
        """Inspect executable for torch tensor usage."""
        # Check type annotations, imports, or a declared flag
        ...
```

After first execution completes, the profiler runs both backends on a small input and caches the faster one for future runs of that executable type.

## Device Routing Flow

```
Executable.device declared?
├── CPU → CPUBackend (OS thread)
├── GPU → MPSBackend (or MetalBackend if MetalKernel)
└── AUTO or not declared
    │
    ├── _uses_tensors() == True → MPSBackend
    ├── isinstance(MetalKernel) → MetalBackend
    └── else → CPUBackend
    │
    └── After first run: profiler validates, caches result
```

## Threading Model

| Actor | Dispatch | Rationale |
|-------|----------|-----------|
| Executor (GPU work) | Async Metal/MPS dispatch from event loop | Metal command buffers are async; synchronize only when results needed |
| Executor (CPU work) | Real OS thread via ThreadPoolExecutor | True parallelism with GIL disabled |
| Observer | Real OS thread | CPU-bound polling, must not block event loop |
| Governor | Real OS thread | CPU-bound policy checks, concurrent with everything |

The event loop thread coordinates lifecycle and messaging. Real work happens on OS threads + GPU hardware simultaneously.

## Changes to Existing Code

### `Executable` base class
- Add optional `device: DevicePreference = DevicePreference.AUTO` class attribute

### `Capabilities` dataclass
- Add `gpu_capable: bool = False` field

### `SentinelRuntime`
- Constructor creates `HybridRunner` with detected backends
- `execute()` delegates to `HybridRunner.run_actors()` instead of `asyncio.gather()`

### `BaseActor`
- Add `run_threaded()` method that wraps `_run()` for execution on an OS thread
- Keep async `start()/stop()` interface for compatibility

## File Structure (new files)

```
sentinal/
├── compute/
│   ├── __init__.py
│   ├── backend.py          # ComputeBackend ABC
│   ├── cpu_backend.py      # CPUBackend (ThreadPoolExecutor)
│   ├── mps_backend.py      # MPSBackend (PyTorch MPS)
│   ├── metal_backend.py    # MetalBackend (raw Metal shaders)
│   ├── device.py           # DevicePreference enum
│   ├── profiler.py         # WorkloadProfiler
│   └── runner.py           # HybridRunner
```

## Testing Strategy

1. **Unit tests**: Each backend in isolation (dispatch, profile, is_available)
2. **Integration test**: HybridRunner routes correctly based on device preference
3. **Parallelism proof**: Extend existing `test_true_parallelism.py` and `test_gpu_parallelism.py` to exercise the runtime (not just raw threads)
4. **Fallback test**: GPU unavailable → graceful fallback to CPU threads
5. **Profiler test**: AUTO routing produces correct cached decisions

## Error Handling

- GPU unavailable at runtime → log warning, fall back to CPUBackend
- Metal shader compilation fails → raise with shader source in error message
- Profiler benchmark fails → default to CPU, log the failure
- Thread dies → actor state transitions to FAILED, Governor can retry

## Future Extensibility

The `ComputeBackend` ABC is designed so adding CUDA or ROCm only requires:
1. New backend class implementing the ABC
2. Register in `HybridRunner` constructor
3. Add detection in startup

No changes to actors, executables, or the runner's routing logic.
