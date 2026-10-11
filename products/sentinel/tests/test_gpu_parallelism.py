"""
Test: GPU Parallelism — Proving Metal GPU Dispatch (Apple Silicon)

Proves that computation actually runs on GPU hardware, and that GPU + CPU
threads can execute simultaneously with GIL disabled. Two backends tested:

1. PyTorch MPS (Metal Performance Shaders) — high-level GPU compute
2. Raw Metal via pyobjc — direct compute shader dispatch

Proof strategies:

A) GPU vs CPU SPEEDUP: Large matrix ops on GPU vastly outperform CPU,
   proving work is offloaded to the GPU's thousands of cores.

B) GPU + CPU OVERLAP: With GIL disabled, CPU threads do real work WHILE
   GPU kernels execute asynchronously — true heterogeneous parallelism.

C) RAW METAL DISPATCH: A hand-written Metal compute shader runs on GPU,
   proving kernel-level GPU thread parallelism (not just library calls).

Usage:
    PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_gpu_parallelism.py -v -s
"""

import sys
import time
import threading

import pytest
import torch

try:
    import Metal
    HAS_METAL = True
except ImportError:
    HAS_METAL = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def check_environment():
    """Ensure we're on Apple Silicon with MPS available."""
    assert sys._is_gil_enabled() is False, "GIL must be disabled (PYTHON_GIL=0)"
    assert torch.backends.mps.is_available(), "MPS not available on this machine"


# ---------------------------------------------------------------------------
# A) GPU vs CPU Speedup
# ---------------------------------------------------------------------------

class TestGPUvsCPUSpeedup:
    """
    PROOF: GPU actually does the work.
    Matrix multiplication on GPU should be dramatically faster than CPU
    for large matrices, proving the Metal GPU cores are being used.
    """

    @pytest.mark.parametrize("size", [2048, 4096, 8192])
    def test_matmul_gpu_faster_than_cpu(self, size):
        """Large matmul should be faster on GPU than CPU. Advantage grows with size."""
        # CPU baseline
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)

        # Warmup CPU
        _ = a_cpu @ b_cpu

        reps = 5
        start = time.perf_counter()
        for _ in range(reps):
            _ = a_cpu @ b_cpu
        cpu_time = (time.perf_counter() - start) / reps

        # GPU (MPS)
        a_gpu = a_cpu.to("mps")
        b_gpu = b_cpu.to("mps")

        # Warmup GPU (first op compiles the shader)
        for _ in range(3):
            _ = a_gpu @ b_gpu
        torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(reps):
            _ = a_gpu @ b_gpu
        torch.mps.synchronize()
        gpu_time = (time.perf_counter() - start) / reps

        speedup = cpu_time / gpu_time

        print(f"\n  Matrix size: {size}x{size}")
        print(f"  CPU time:    {cpu_time*1000:.1f}ms")
        print(f"  GPU time:    {gpu_time*1000:.1f}ms")
        print(f"  GPU speedup: {speedup:.1f}x")

        # M1's CPU has excellent NEON SIMD, so advantage is smaller at small sizes.
        # At 8192x8192 the GPU's parallelism dominates clearly.
        min_speedup = {2048: 1.3, 4096: 1.5, 8192: 1.7}[size]
        assert speedup > min_speedup, (
            f"GPU speedup only {speedup:.1f}x for {size}x{size} matmul — "
            f"expected >{min_speedup}x. Work may not be running on GPU."
        )

    def test_batch_matmul_gpu_throughput(self):
        """Batch of many matmuls — GPU's parallelism shines on repeated dispatch."""
        size = 1024
        batch = 50

        # CPU
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        _ = a_cpu @ b_cpu  # warmup

        start = time.perf_counter()
        for _ in range(batch):
            _ = a_cpu @ b_cpu
        cpu_time = time.perf_counter() - start

        # GPU
        a_gpu = a_cpu.to("mps")
        b_gpu = b_cpu.to("mps")
        for _ in range(3):
            _ = a_gpu @ b_gpu
        torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(batch):
            _ = a_gpu @ b_gpu
        torch.mps.synchronize()
        gpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time

        print(f"\n  Batch matmul ({batch}x {size}x{size}):")
        print(f"  CPU: {cpu_time*1000:.1f}ms  GPU: {gpu_time*1000:.1f}ms")
        print(f"  Speedup: {speedup:.1f}x")

        assert speedup > 1.3, (
            f"GPU batch matmul speedup only {speedup:.1f}x — "
            f"kernels may not be pipelining on Metal"
        )


# ---------------------------------------------------------------------------
# B) GPU + CPU Thread Overlap (Heterogeneous Parallelism)
# ---------------------------------------------------------------------------

class TestGPUCPUOverlap:
    """
    PROOF: GPU and CPU work happen SIMULTANEOUSLY.
    With GIL disabled, CPU threads can do real computation while GPU
    kernels are in flight. This is true heterogeneous parallelism.
    """

    def test_gpu_and_cpu_threads_overlap(self):
        """
        Launch GPU work (async) and CPU work (in thread) simultaneously.
        Total wall time should be ~max(gpu, cpu), not gpu + cpu.
        """
        size = 4096
        cpu_iterations = 5_000_000

        # Prepare GPU tensors
        a_gpu = torch.randn(size, size, device="mps")
        b_gpu = torch.randn(size, size, device="mps")
        # Warmup
        _ = a_gpu @ b_gpu
        torch.mps.synchronize()

        # Measure GPU-only time
        start = time.perf_counter()
        for _ in range(5):
            _ = a_gpu @ b_gpu
        torch.mps.synchronize()
        gpu_only_time = time.perf_counter() - start

        # Measure CPU-only time (pure computation, no GIL release)
        def cpu_work():
            total = 0.0
            for i in range(cpu_iterations):
                total += i * i * 0.000001
            return total

        start = time.perf_counter()
        cpu_work()
        cpu_only_time = time.perf_counter() - start

        # Now run BOTH simultaneously
        cpu_result = [None]
        cpu_window = [None, None]

        def cpu_thread_fn():
            cpu_window[0] = time.perf_counter()
            cpu_result[0] = cpu_work()
            cpu_window[1] = time.perf_counter()

        start = time.perf_counter()

        # Start CPU thread
        cpu_thread = threading.Thread(target=cpu_thread_fn)
        cpu_thread.start()

        # Dispatch GPU work (async on Metal)
        gpu_start = time.perf_counter()
        for _ in range(5):
            _ = a_gpu @ b_gpu
        torch.mps.synchronize()
        gpu_end = time.perf_counter()

        cpu_thread.join()
        combined_time = time.perf_counter() - start

        serial_time = gpu_only_time + cpu_only_time

        # Check overlap: CPU thread's window should overlap with GPU window
        gpu_window = (gpu_start, gpu_end)
        overlap = min(gpu_window[1], cpu_window[1]) - max(gpu_window[0], cpu_window[0])

        print(f"\n  GPU-only: {gpu_only_time*1000:.1f}ms")
        print(f"  CPU-only: {cpu_only_time*1000:.1f}ms")
        print(f"  Combined: {combined_time*1000:.1f}ms")
        print(f"  Serial would be: {serial_time*1000:.1f}ms")
        print(f"  Overlap: {overlap*1000:.1f}ms")
        print(f"  Efficiency: {serial_time/combined_time:.2f}x (1.0=no overlap, 2.0=perfect)")

        # Combined time should be significantly less than serial
        assert combined_time < serial_time * 0.85, (
            f"Combined ({combined_time*1000:.0f}ms) not faster than serial "
            f"({serial_time*1000:.0f}ms) — GPU and CPU not overlapping"
        )

        assert overlap > 0, "GPU and CPU execution windows did not overlap at all"

    def test_multiple_cpu_threads_while_gpu_works(self):
        """
        Multiple CPU threads + GPU all executing simultaneously.
        Proves the free-threaded runtime enables full heterogeneous parallelism.
        """
        num_cpu_threads = 4
        cpu_iters = 3_000_000
        size = 4096

        a_gpu = torch.randn(size, size, device="mps")
        b_gpu = torch.randn(size, size, device="mps")
        _ = a_gpu @ b_gpu
        torch.mps.synchronize()

        windows = {}

        def cpu_work_tracked(name):
            t0 = time.perf_counter()
            total = 0.0
            for i in range(cpu_iters):
                total += i * i * 0.000001
            t1 = time.perf_counter()
            windows[name] = (t0, t1)

        barrier = threading.Barrier(num_cpu_threads + 1)  # +1 for main/GPU

        def cpu_with_barrier(name):
            barrier.wait()
            cpu_work_tracked(name)

        # Launch CPU threads
        threads = []
        for i in range(num_cpu_threads):
            t = threading.Thread(target=cpu_with_barrier, args=(f"cpu_{i}",))
            threads.append(t)
            t.start()

        # Main thread does GPU work after barrier
        barrier.wait()
        gpu_t0 = time.perf_counter()
        for _ in range(10):
            _ = a_gpu @ b_gpu
        torch.mps.synchronize()
        gpu_t1 = time.perf_counter()
        windows["gpu"] = (gpu_t0, gpu_t1)

        for t in threads:
            t.join()

        # Check that GPU overlaps with CPU threads
        gpu_start, gpu_end = windows["gpu"]
        overlapping_with_gpu = 0
        for name, (t0, t1) in windows.items():
            if name == "gpu":
                continue
            if t0 < gpu_end and gpu_start < t1:
                overlapping_with_gpu += 1

        print(f"\n  Heterogeneous parallelism ({num_cpu_threads} CPU + 1 GPU):")
        for name, (t0, t1) in sorted(windows.items()):
            print(f"    {name:6s}: {(t1-t0)*1000:.1f}ms")
        print(f"  CPU threads overlapping with GPU: {overlapping_with_gpu}/{num_cpu_threads}")

        assert overlapping_with_gpu == num_cpu_threads, (
            f"Only {overlapping_with_gpu}/{num_cpu_threads} CPU threads overlapped "
            f"with GPU execution — not fully parallel"
        )


# ---------------------------------------------------------------------------
# C) Raw Metal Compute Shader
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_METAL, reason="pyobjc-framework-Metal not installed")
class TestRawMetalDispatch:
    """
    PROOF: Direct Metal compute shader execution.
    Write a kernel, dispatch it to GPU, verify results — proving that
    GPU threads (SIMD lanes) execute in parallel at the hardware level.
    """

    def test_metal_vector_add_kernel(self):
        """
        Dispatch a Metal compute shader that adds two arrays element-wise.
        Verifies GPU threads execute independently and produce correct results.
        """
        import struct

        device = Metal.MTLCreateSystemDefaultDevice()
        assert device is not None, "No Metal device found"

        shader_src = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void vector_add(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* result [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            result[id] = a[id] + b[id];
        }
        """

        n = 1_000_000  # 1M elements — 1M GPU threads

        # Compile shader
        options = Metal.MTLCompileOptions.alloc().init()
        library, error = device.newLibraryWithSource_options_error_(
            shader_src, options, None
        )
        assert library is not None, f"Shader compile failed: {error}"

        function = library.newFunctionWithName_("vector_add")
        assert function is not None, "Could not find vector_add function"

        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        assert pipeline is not None, f"Pipeline creation failed: {error}"

        # Allocate GPU buffers (shared memory — accessible from CPU and GPU)
        float_size = 4
        buf_size = n * float_size

        buf_a = device.newBufferWithLength_options_(buf_size, Metal.MTLResourceStorageModeShared)
        buf_b = device.newBufferWithLength_options_(buf_size, Metal.MTLResourceStorageModeShared)
        buf_result = device.newBufferWithLength_options_(buf_size, Metal.MTLResourceStorageModeShared)

        # Fill input buffers via memoryview
        mv_a = memoryview(buf_a.contents().as_buffer(buf_size)).cast('f')
        mv_b = memoryview(buf_b.contents().as_buffer(buf_size)).cast('f')

        for i in range(n):
            mv_a[i] = float(i)
            mv_b[i] = float(i * 2)

        # Dispatch compute shader
        command_queue = device.newCommandQueue()
        command_buffer = command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_result, 0, 2)

        threads_per_group = min(pipeline.maxTotalThreadsPerThreadgroup(), 256)
        grid_size = Metal.MTLSizeMake(n, 1, 1)
        group_size = Metal.MTLSizeMake(threads_per_group, 1, 1)

        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, group_size)
        encoder.endEncoding()

        start = time.perf_counter()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        gpu_time = time.perf_counter() - start

        # Verify results
        mv_result = memoryview(buf_result.contents().as_buffer(buf_size)).cast('f')

        errors = 0
        spots = [0, 1, 100, 999, n // 2, n - 1]
        for i in spots:
            expected = float(i) + float(i * 2)
            actual = mv_result[i]
            if abs(actual - expected) > 0.01:
                errors += 1

        # CPU comparison for timing
        start = time.perf_counter()
        result_cpu = [float(i) + float(i * 2) for i in range(n)]
        cpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time

        print(f"\n  Raw Metal Compute Shader:")
        print(f"  GPU: {device.name()}")
        print(f"  Elements: {n:,}")
        print(f"  GPU threads dispatched: {n:,}")
        print(f"  Threads per group: {threads_per_group}")
        print(f"  GPU time: {gpu_time*1000:.2f}ms")
        print(f"  CPU time: {cpu_time*1000:.1f}ms")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Verification errors: {errors}")

        assert errors == 0, f"GPU produced wrong results at {errors} positions"
        assert speedup > 2.0, (
            f"Metal shader speedup only {speedup:.1f}x — "
            f"kernel may not be utilizing GPU parallelism"
        )

    def test_metal_parallel_reduction(self):
        """
        A reduction kernel that proves GPU threads cooperate in parallel.
        Each threadgroup reduces a chunk, then we sum the partial results.
        The threadgroup_barrier ONLY works if threads run simultaneously —
        if they didn't, the barrier would deadlock.
        """
        device = Metal.MTLCreateSystemDefaultDevice()

        shader_src = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void parallel_sum(
            device const float* input [[buffer(0)]],
            device float* partials [[buffer(1)]],
            uint gid [[thread_position_in_grid]],
            uint lid [[thread_position_in_threadgroup]],
            uint group_id [[threadgroup_position_in_grid]],
            uint group_size [[threads_per_threadgroup]]
        ) {
            threadgroup float shared[256];
            shared[lid] = input[gid];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    shared[lid] += shared[lid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                partials[group_id] = shared[0];
            }
        }
        """

        n = 256 * 4096
        group_size_val = 256
        num_groups = n // group_size_val

        options = Metal.MTLCompileOptions.alloc().init()
        library, error = device.newLibraryWithSource_options_error_(
            shader_src, options, None
        )
        assert library is not None, f"Compile failed: {error}"

        function = library.newFunctionWithName_("parallel_sum")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        assert pipeline is not None, f"Pipeline failed: {error}"

        float_size = 4
        buf_input = device.newBufferWithLength_options_(
            n * float_size, Metal.MTLResourceStorageModeShared
        )
        buf_partials = device.newBufferWithLength_options_(
            num_groups * float_size, Metal.MTLResourceStorageModeShared
        )

        # Fill with 1.0 — expected sum is n
        mv_input = memoryview(buf_input.contents().as_buffer(n * float_size)).cast('f')
        for i in range(n):
            mv_input[i] = 1.0

        command_queue = device.newCommandQueue()
        command_buffer = command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(buf_input, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_partials, 0, 1)

        grid = Metal.MTLSizeMake(n, 1, 1)
        group = Metal.MTLSizeMake(group_size_val, 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid, group)
        encoder.endEncoding()

        start = time.perf_counter()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        gpu_time = time.perf_counter() - start

        # Read partial sums
        mv_partials = memoryview(
            buf_partials.contents().as_buffer(num_groups * float_size)
        ).cast('f')
        total = sum(mv_partials[i] for i in range(num_groups))

        print(f"\n  Parallel Reduction (threadgroup cooperation):")
        print(f"  Elements: {n:,}")
        print(f"  Threadgroups: {num_groups}")
        print(f"  Threads per group: {group_size_val}")
        print(f"  GPU time: {gpu_time*1000:.2f}ms")
        print(f"  Expected sum: {n:.1f}")
        print(f"  Actual sum:   {total:.1f}")

        assert abs(total - float(n)) < n * 0.001, (
            f"Reduction result {total} != expected {n} — "
            f"threadgroup_barrier may not be working (threads not parallel)"
        )


# ---------------------------------------------------------------------------
# D) Combined: Sentinel Actors on CPU Threads + GPU Acceleration
# ---------------------------------------------------------------------------

class TestSentinelWithGPU:
    """
    The real-world payoff: Sentinel actors running on separate OS threads
    while one of them offloads heavy work to GPU — all truly simultaneous.
    """

    def test_actor_threads_with_gpu_offload(self):
        """
        Simulate the 3-thread Sentinel model where:
        - Executor: offloads computation to GPU
        - Observer: CPU-bound analysis on a separate core
        - Governor: CPU-bound policy checks on another core

        All three should overlap in time.
        """
        size = 4096
        cpu_iters = 3_000_000
        results = {}
        barrier = threading.Barrier(3)

        def executor_with_gpu():
            """Actor that dispatches work to GPU."""
            a = torch.randn(size, size, device="mps")
            b = torch.randn(size, size, device="mps")
            _ = a @ b
            torch.mps.synchronize()

            barrier.wait()
            tid = threading.current_thread().native_id
            t0 = time.perf_counter()
            for _ in range(10):
                _ = a @ b
            torch.mps.synchronize()
            t1 = time.perf_counter()
            results["executor(GPU)"] = {"tid": tid, "start": t0, "end": t1}

        def observer_cpu():
            """Actor doing CPU-bound observation."""
            barrier.wait()
            tid = threading.current_thread().native_id
            t0 = time.perf_counter()
            total = 0.0
            for i in range(cpu_iters):
                total += i * i * 0.000001
            t1 = time.perf_counter()
            results["observer(CPU)"] = {"tid": tid, "start": t0, "end": t1}

        def governor_cpu():
            """Actor doing CPU-bound governance."""
            barrier.wait()
            tid = threading.current_thread().native_id
            t0 = time.perf_counter()
            total = 0.0
            for i in range(cpu_iters):
                total += (i + 1) ** 0.5
            t1 = time.perf_counter()
            results["governor(CPU)"] = {"tid": tid, "start": t0, "end": t1}

        threads = [
            threading.Thread(target=executor_with_gpu, name="executor"),
            threading.Thread(target=observer_cpu, name="observer"),
            threading.Thread(target=governor_cpu, name="governor"),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all on different OS threads
        tids = [r["tid"] for r in results.values()]
        assert len(set(tids)) == 3, f"Actors not on separate OS threads: {tids}"

        # Verify all overlap pairwise
        actors = list(results.keys())
        all_overlap = True
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                a = results[actors[i]]
                b = results[actors[j]]
                if not (a["start"] < b["end"] and b["start"] < a["end"]):
                    all_overlap = False

        wall_start = min(r["start"] for r in results.values())
        wall_end = max(r["end"] for r in results.values())
        wall_time = wall_end - wall_start
        sum_time = sum(r["end"] - r["start"] for r in results.values())

        print(f"\n  Sentinel Actors — Heterogeneous Parallelism:")
        print(f"  {'Actor':<18} {'TID':<10} {'Duration':>10}")
        print(f"  {'-'*40}")
        for name, r in results.items():
            dur = (r["end"] - r["start"]) * 1000
            print(f"  {name:<18} {r['tid']:<10} {dur:>8.1f}ms")
        print(f"  {'-'*40}")
        print(f"  Wall time:  {wall_time*1000:.1f}ms")
        print(f"  Sum (if serial): {sum_time*1000:.1f}ms")
        print(f"  Parallelism efficiency: {sum_time/wall_time:.2f}x")

        assert all_overlap, (
            "Not all actor pairs overlapped — actors are not running in parallel"
        )
        assert wall_time < sum_time * 0.7, (
            f"Wall time ({wall_time*1000:.0f}ms) too close to serial sum "
            f"({sum_time*1000:.0f}ms) — insufficient parallelism"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])