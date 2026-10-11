"""
Test: True OS-Thread Parallelism with GIL Disabled (Python 3.14t)

This test proves that threads execute SIMULTANEOUSLY on separate CPU cores,
not just concurrently (interleaved on one core). Three proof strategies:

1. TIMING PROOF: CPU-bound work in N threads completes in ~1/N wall time
   vs sequential. With GIL, threads serialize → no speedup.

2. OVERLAP PROOF: Threads record timestamps while running. With true
   parallelism, their execution windows overlap. With GIL, they don't.

3. CONTENTION PROOF: Unsynchronized increments on a shared counter produce
   a lost-update race condition ONLY if threads truly run simultaneously.
   With GIL, every inc is atomic → no lost updates → no race.

Usage:
    PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_true_parallelism.py -v
"""

import sys
import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor

import pytest


NUM_THREADS = 4
WORK_PER_THREAD = 5_000_000  # iterations of pure CPU work


def cpu_bound_work(n: int) -> float:
    """Pure CPU-bound computation — no I/O, no allocations that could yield."""
    total = 0.0
    for i in range(n):
        total += i * i * 0.000001
    return total


class TestGILDisabled:
    """Verify the environment is correctly configured."""

    def test_python_version_is_free_threaded(self):
        assert sys.version_info >= (3, 14), (
            f"Need Python 3.14t, got {sys.version}"
        )

    def test_gil_is_disabled(self):
        assert hasattr(sys, "_is_gil_enabled"), (
            "Not a free-threaded build — sys._is_gil_enabled missing"
        )
        assert sys._is_gil_enabled() is False, (
            "GIL is enabled! Set PYTHON_GIL=0 or use python3.14t"
        )


class TestTimingProof:
    """
    PROOF BY TIMING:
    If threads truly run on separate OS cores, N threads doing W work each
    should take ~W time total (parallel), not ~N*W (serial).

    We require at least 2x speedup with 4 threads. With GIL, speedup ≈ 1.0.
    """

    def test_parallel_speedup_over_sequential(self):
        # Sequential baseline: run all work on one thread
        start = time.perf_counter()
        for _ in range(NUM_THREADS):
            cpu_bound_work(WORK_PER_THREAD)
        sequential_time = time.perf_counter() - start

        # Parallel: distribute across OS threads
        start = time.perf_counter()
        threads = []
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=cpu_bound_work, args=(WORK_PER_THREAD,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        parallel_time = time.perf_counter() - start

        speedup = sequential_time / parallel_time

        print(f"\n  Sequential: {sequential_time:.3f}s")
        print(f"  Parallel:   {parallel_time:.3f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        print(f"  Threads:    {NUM_THREADS}")

        # With GIL disabled + multiple cores, expect >2x speedup with 4 threads.
        # With GIL enabled, speedup would be ~0.9-1.1x (often slower due to contention).
        assert speedup > 2.0, (
            f"Speedup was only {speedup:.2f}x — threads are NOT running in parallel. "
            f"GIL may still be active or only 1 CPU core available."
        )


class TestOverlapProof:
    """
    PROOF BY OVERLAP:
    Each thread records [start_time, end_time] while doing CPU work.
    If truly parallel, time windows overlap. If GIL-serialized, they don't.
    """

    def test_thread_execution_windows_overlap(self):
        windows = [None] * NUM_THREADS
        thread_ids = [None] * NUM_THREADS

        def timed_work(idx):
            thread_ids[idx] = threading.current_thread().native_id
            t0 = time.perf_counter()
            cpu_bound_work(WORK_PER_THREAD)
            t1 = time.perf_counter()
            windows[idx] = (t0, t1)

        threads = []
        for i in range(NUM_THREADS):
            t = threading.Thread(target=timed_work, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # All threads should have different OS thread IDs
        unique_ids = set(thread_ids)
        assert len(unique_ids) == NUM_THREADS, (
            f"Expected {NUM_THREADS} unique OS thread IDs, got {len(unique_ids)}: {thread_ids}"
        )

        # Check for pairwise overlap: two windows [a0,a1] and [b0,b1] overlap
        # if a0 < b1 AND b0 < a1
        overlap_count = 0
        for i in range(NUM_THREADS):
            for j in range(i + 1, NUM_THREADS):
                a0, a1 = windows[i]
                b0, b1 = windows[j]
                if a0 < b1 and b0 < a1:
                    overlap_count += 1

        total_pairs = NUM_THREADS * (NUM_THREADS - 1) // 2

        print(f"\n  Thread IDs (OS-level): {list(unique_ids)}")
        print(f"  Overlapping pairs: {overlap_count}/{total_pairs}")
        for i, (t0, t1) in enumerate(windows):
            print(f"  Thread {i}: {t0:.6f} - {t1:.6f} ({(t1-t0)*1000:.1f}ms)")

        # With true parallelism, ALL pairs should overlap (they all start ~simultaneously)
        # With GIL, zero pairs overlap (fully serialized)
        assert overlap_count == total_pairs, (
            f"Only {overlap_count}/{total_pairs} thread pairs overlapped — "
            f"threads are being serialized (GIL likely active)"
        )


class TestContentionProof:
    """
    PROOF BY RACE CONDITION:
    With GIL, `counter += 1` is effectively atomic (bytecode-level lock).
    Without GIL, concurrent increments produce lost updates.

    If we observe lost updates → threads truly ran simultaneously on different cores.
    """

    def test_shared_counter_race_condition(self):
        # Use a mutable container to avoid thread-local rebinding
        shared = [0]
        iterations = 1_000_000

        def increment_shared():
            for _ in range(iterations):
                shared[0] += 1  # NOT atomic without GIL

        threads = []
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=increment_shared)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        expected = NUM_THREADS * iterations
        actual = shared[0]
        lost_updates = expected - actual

        print(f"\n  Expected: {expected:,}")
        print(f"  Actual:   {actual:,}")
        print(f"  Lost:     {lost_updates:,} ({lost_updates/expected*100:.1f}%)")

        # With GIL: actual == expected (no race, += is atomic under GIL)
        # Without GIL: actual < expected (race condition, lost updates)
        assert actual < expected, (
            f"No lost updates detected (counter={actual}, expected={expected}). "
            f"This means += is still atomic — GIL may still be protecting operations. "
            f"Threads are NOT truly racing."
        )


class TestThreadPoolParallelism:
    """
    Verify ThreadPoolExecutor also achieves true parallelism.
    This is the pattern your runtime would use for real workloads.
    """

    def test_threadpool_achieves_parallel_speedup(self):
        # Sequential
        start = time.perf_counter()
        results_seq = [cpu_bound_work(WORK_PER_THREAD) for _ in range(NUM_THREADS)]
        sequential_time = time.perf_counter() - start

        # Parallel via ThreadPoolExecutor
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(cpu_bound_work, WORK_PER_THREAD) for _ in range(NUM_THREADS)]
            results_par = [f.result() for f in futures]
        parallel_time = time.perf_counter() - start

        speedup = sequential_time / parallel_time

        print(f"\n  ThreadPoolExecutor speedup: {speedup:.2f}x")

        assert speedup > 2.0, (
            f"ThreadPoolExecutor speedup was only {speedup:.2f}x"
        )

        # Results should be numerically identical regardless of parallelism
        for s, p in zip(results_seq, results_par):
            assert abs(s - p) < 1e-6


class TestSentinelRuntimeParallelism:
    """
    Test that Sentinel's actor model can leverage true thread parallelism
    for CPU-bound supervision tasks.
    """

    def test_actors_run_on_separate_os_threads(self):
        """Simulate 3 sentinel actors (executor, observer, governor) on real threads."""
        results = {}
        barrier = threading.Barrier(3)  # sync start for clean measurement

        def actor_work(name: str):
            tid = threading.current_thread().native_id
            barrier.wait()  # all start together
            t0 = time.perf_counter()
            cpu_bound_work(WORK_PER_THREAD)
            t1 = time.perf_counter()
            results[name] = {
                "thread_id": tid,
                "start": t0,
                "end": t1,
                "duration_ms": (t1 - t0) * 1000,
            }

        actors = ["executor", "observer", "governor"]
        threads = []
        for name in actors:
            t = threading.Thread(target=actor_work, args=(name,), name=name)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # Verify different OS threads
        tids = [results[a]["thread_id"] for a in actors]
        assert len(set(tids)) == 3, f"Actors share threads: {tids}"

        # Verify overlap (all should overlap since barrier-synced start)
        for i, a in enumerate(actors):
            for j, b in enumerate(actors):
                if i >= j:
                    continue
                ra, rb = results[a], results[b]
                assert ra["start"] < rb["end"] and rb["start"] < ra["end"], (
                    f"{a} and {b} did NOT execute in parallel"
                )

        # Wall time should be ~1 unit of work, not 3
        wall_start = min(r["start"] for r in results.values())
        wall_end = max(r["end"] for r in results.values())
        wall_time = wall_end - wall_start
        single_time = max(r["duration_ms"] for r in results.values())

        print(f"\n  Sentinel Actor Parallelism:")
        for name in actors:
            r = results[name]
            print(f"    {name}: tid={r['thread_id']} duration={r['duration_ms']:.1f}ms")
        print(f"  Wall time: {wall_time*1000:.1f}ms (vs {single_time*3:.1f}ms if serial)")

        # 3 actors in parallel → wall time should be < 2x single actor time
        assert wall_time * 1000 < single_time * 2, (
            f"Wall time {wall_time*1000:.1f}ms >= 2x single actor {single_time:.1f}ms — "
            "actors are not running in parallel"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
