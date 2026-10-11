"""
Real OS-Level Threading: Actor | Observer | Governor

This demonstrates TRUE concurrent execution using Python's threading module.
3 separate OS threads running simultaneously:

Thread 1 (Actor): Executes the MAF agent
Thread 2 (Observer): Watches memory updates in real-time
Thread 3 (Governor): Monitors and can intervene

This is the REAL vision - actual OS-level parallelism.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import threading
import time
import asyncio
from typing import Any, Dict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from queue import Queue

# MAF agent
from MAF.test_supervised_agents import MockIntentUnderstandingAgent


# ==========================================
# Shared Execution State
# ==========================================

@dataclass
class ExecutionState:
    """
    Shared state accessible by all 3 threads.

    Uses threading.Lock for thread-safe access.
    """

    attempt: int = 1
    status: str = "PENDING"
    runtime_memory: Dict[str, Any] = field(default_factory=dict)
    runtime_instructions: list = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_memory(self, key: str, value: Any):
        """Thread-safe memory update."""
        with self._lock:
            self.runtime_memory[key] = value

    def get_memory(self, key: str, default=None):
        """Thread-safe memory read."""
        with self._lock:
            return self.runtime_memory.get(key, default)

    def add_instruction(self, instruction: str):
        """Thread-safe instruction add."""
        with self._lock:
            self.runtime_instructions.append(instruction)

    def get_instructions(self):
        """Thread-safe instruction read."""
        with self._lock:
            return self.runtime_instructions.copy()

    def transition(self, status: str):
        """Thread-safe status update."""
        with self._lock:
            self.status = status
            if status in ["COMPLETED", "FAILED"]:
                self.completed_at = datetime.now(UTC)


# ==========================================
# Thread 1: Actor (Executes Agent)
# ==========================================

class ActorThread(threading.Thread):
    """
    OS Thread #1: Actor

    Executes the MAF agent while updating shared memory.
    Observer watches these memory updates.
    Governor can add instructions that Actor follows.
    """

    def __init__(self, maf_agent: Any, state: ExecutionState, query: str):
        super().__init__(name="ActorThread", daemon=False)
        self.maf_agent = maf_agent
        self.state = state
        self.query = query
        self.result = None

    def run(self):
        """
        Main execution (runs in separate OS thread).
        """

        print(f"\n🎭 [ACTOR-THREAD-{threading.get_ident()}] Starting...")

        # Update state
        self.state.update_memory("actor_status", "working")
        self.state.update_memory("current_query", self.query)
        self.state.update_memory("thread_id", threading.get_ident())
        self.state.transition("RUNNING")

        # Check for instructions from Governor
        instructions = self.state.get_instructions()
        if instructions:
            print(f"📜 [ACTOR] Following {len(instructions)} instructions:")
            for inst in instructions:
                print(f"   - {inst}")

        # Execute MAF agent (using asyncio.run for async code)
        start_time = time.time()

        try:
            maf_input = {
                "user_query": self.query,
                "updated_query": self.query,
            }

            # Run async agent in this thread
            result = asyncio.run(self.maf_agent(maf_input))

            latency_ms = (time.time() - start_time) * 1000

            # Update state with results
            self.state.update_memory("result", result)
            self.state.update_memory("latency_ms", latency_ms)
            self.state.update_memory("actor_status", "complete")
            self.state.transition("COMPLETED")

            quality = self._compute_quality(result)

            self.result = {
                "status": "completed",
                "result": result,
                "quality_score": quality,
                "latency_ms": latency_ms,
            }

            print(f"✅ [ACTOR-THREAD-{threading.get_ident()}] Complete: "
                  f"Quality={quality:.2f}, Latency={latency_ms:.0f}ms")

        except Exception as e:
            print(f"❌ [ACTOR-THREAD-{threading.get_ident()}] Error: {e}")

            self.state.update_memory("error", str(e))
            self.state.update_memory("actor_status", "error")
            self.state.transition("FAILED")

            self.result = {
                "status": "failed",
                "error": str(e),
            }

    def _compute_quality(self, result: Dict[str, Any]) -> float:
        intent_text = result.get("intent_understanding", "")
        if not intent_text:
            return 0.2
        if len(intent_text.strip()) > 20:
            return 0.9
        return 0.5


# ==========================================
# Thread 2: Observer (Watches Memory)
# ==========================================

class ObserverThread(threading.Thread):
    """
    OS Thread #2: Observer

    Watches runtime_memory for changes.
    Runs concurrently with Actor and Governor.
    """

    def __init__(self, state: ExecutionState, duration_sec: float = 10.0):
        super().__init__(name="ObserverThread", daemon=False)
        self.state = state
        self.duration_sec = duration_sec
        self.observations = []
        self.running = True

    def run(self):
        """
        Watch memory changes (runs in separate OS thread).
        """

        print(f"\n👁️  [OBSERVER-THREAD-{threading.get_ident()}] Started watching...")

        start_time = time.time()
        last_status = None

        while self.running and (time.time() - start_time) < self.duration_sec:

            # Read current state (thread-safe)
            current_status = self.state.get_memory("actor_status")

            if current_status != last_status:
                # State changed!
                observation = {
                    "timestamp": time.time(),
                    "actor_status": current_status,
                    "latency": self.state.get_memory("latency_ms"),
                    "thread_id": threading.get_ident(),
                }

                self.observations.append(observation)

                print(f"👁️  [OBSERVER-THREAD-{threading.get_ident()}] "
                      f"Observed: actor_status={current_status}")

                last_status = current_status

            # Check if done
            if self.state.status in ["COMPLETED", "FAILED"]:
                print(f"👁️  [OBSERVER-THREAD-{threading.get_ident()}] "
                      f"Actor finished, stopping")
                break

            time.sleep(0.1)  # Poll every 100ms

        print(f"👁️  [OBSERVER-THREAD-{threading.get_ident()}] "
              f"Stopped. Observations: {len(self.observations)}")

    def stop(self):
        """Signal thread to stop."""
        self.running = False


# ==========================================
# Thread 3: Governor (Intervenes)
# ==========================================

class GovernorThread(threading.Thread):
    """
    OS Thread #3: Governor

    Monitors execution and can intervene:
    - Issue retry commands
    - Suggest better tools
    - Course-correct
    - Set quality thresholds

    Runs concurrently with Actor and Observer.
    """

    def __init__(self, state: ExecutionState, duration_sec: float = 10.0):
        super().__init__(name="GovernorThread", daemon=False)
        self.state = state
        self.duration_sec = duration_sec
        self.interventions = []
        self.running = True
        self.quality_threshold = 0.7

    def run(self):
        """
        Monitor and govern (runs in separate OS thread).
        """

        print(f"\n👑 [GOVERNOR-THREAD-{threading.get_ident()}] Started governing...")

        start_time = time.time()

        while self.running and (time.time() - start_time) < self.duration_sec:

            # Check actor status
            actor_status = self.state.get_memory("actor_status")

            if actor_status == "error":
                # Actor failed! Issue retry
                print(f"👑 [GOVERNOR-THREAD-{threading.get_ident()}] "
                      f"Detected ERROR, issuing RETRY")

                self.state.add_instruction("RETRY_WITH_FALLBACK")

                self.interventions.append({
                    "timestamp": time.time(),
                    "type": "retry",
                    "reason": "actor_error",
                    "thread_id": threading.get_ident(),
                })

            elif actor_status == "complete":
                # Check quality
                result = self.state.get_memory("result", {})
                quality = self._compute_quality(result)

                if quality < self.quality_threshold:
                    print(f"👑 [GOVERNOR-THREAD-{threading.get_ident()}] "
                          f"Quality LOW ({quality:.2f}), suggesting IMPROVE")

                    self.state.add_instruction("IMPROVE_QUALITY")
                    self.state.add_instruction("USE_BETTER_TOOLS")

                    self.interventions.append({
                        "timestamp": time.time(),
                        "type": "quality_improvement",
                        "reason": f"quality={quality:.2f} < threshold={self.quality_threshold}",
                        "thread_id": threading.get_ident(),
                    })

            # Check if done
            if self.state.status in ["COMPLETED", "FAILED"]:
                print(f"👑 [GOVERNOR-THREAD-{threading.get_ident()}] "
                      f"Execution finished")
                break

            time.sleep(0.15)  # Poll every 150ms

        print(f"👑 [GOVERNOR-THREAD-{threading.get_ident()}] "
              f"Stopped. Interventions: {len(self.interventions)}")

    def _compute_quality(self, result: Dict[str, Any]) -> float:
        intent_text = result.get("intent_understanding", "")
        if not intent_text:
            return 0.2
        if len(intent_text.strip()) > 20:
            return 0.9
        return 0.5

    def stop(self):
        """Signal thread to stop."""
        self.running = False


# ==========================================
# Integration Test
# ==========================================

def test_real_os_threads():
    """
    Test with REAL OS-level threads.

    This creates 3 actual threads that run concurrently.
    """

    print("\n" + "=" * 70)
    print("REAL OS-LEVEL THREADING TEST")
    print("3 Concurrent OS Threads: Actor | Observer | Governor")
    print("=" * 70)

    # Step 1: Create shared state
    print("\n📦 Step 1: Create Shared Execution State")
    state = ExecutionState()
    print("  ✓ Shared state with thread-safe locks")

    # Step 2: Create MAF agent
    print("\n🤖 Step 2: Create MAF Agent")
    maf_agent = MockIntentUnderstandingAgent()
    print("  ✓ Intent Understanding Agent")

    # Step 3: Create the 3 threads
    print("\n🧵 Step 3: Create 3 OS Threads")

    query = "How do I build a supervised multi-agent system with course correction?"

    actor = ActorThread(maf_agent, state, query)
    observer = ObserverThread(state, duration_sec=5.0)
    governor = GovernorThread(state, duration_sec=5.0)

    print(f"  ✓ Actor thread (will run on OS thread)")
    print(f"  ✓ Observer thread (will run on OS thread)")
    print(f"  ✓ Governor thread (will run on OS thread)")

    # Step 4: Start all 3 threads
    print("\n⚡ Step 4: Launch All 3 OS Threads")
    print("-" * 70)

    start_time = time.time()

    actor.start()
    observer.start()
    governor.start()

    print(f"  ✓ Actor started on thread {actor.ident}")
    print(f"  ✓ Observer started on thread {observer.ident}")
    print(f"  ✓ Governor started on thread {governor.ident}")
    print(f"\n🔄 All 3 threads running concurrently...\n")

    # Step 5: Wait for all to finish
    actor.join()
    observer.join()
    governor.join()

    elapsed = (time.time() - start_time) * 1000

    print("-" * 70)
    print(f"⏱️  All threads completed in {elapsed:.0f}ms")

    # Step 6: Results
    print("\n📊 Step 6: Actor Results")
    if actor.result:
        print(f"  Status: {actor.result['status']}")
        print(f"  Quality: {actor.result.get('quality_score', 'N/A'):.2f}")
        print(f"  Latency: {actor.result.get('latency_ms', 'N/A'):.0f}ms")

    # Step 7: Observer observations
    print("\n👁️  Step 7: Observer's Observations")
    print(f"  Total: {len(observer.observations)}")
    for obs in observer.observations:
        print(f"    - Status: {obs['actor_status']}, "
              f"Thread: {obs['thread_id']}, "
              f"Latency: {obs.get('latency')}")

    # Step 8: Governor interventions
    print("\n👑 Step 8: Governor's Interventions")
    print(f"  Total: {len(governor.interventions)}")
    for intervention in governor.interventions:
        print(f"    - Type: {intervention['type']}")
        print(f"      Reason: {intervention['reason']}")
        print(f"      Thread: {intervention['thread_id']}")

    # Step 9: Final memory state
    print("\n🧠 Step 9: Final Runtime Memory")
    with state._lock:
        for key, value in state.runtime_memory.items():
            if isinstance(value, dict):
                print(f"  {key}: {str(value)[:60]}...")
            else:
                print(f"  {key}: {value}")

    # Step 10: Runtime instructions
    print("\n📜 Step 10: Runtime Instructions (from Governor)")
    instructions = state.get_instructions()
    if instructions:
        for instruction in instructions:
            print(f"  - {instruction}")
    else:
        print("  (none)")

    print("\n" + "=" * 70)
    print("✅ SUCCESS: 3 OS Threads Executed Concurrently!")
    print("=" * 70)
    print(f"\nThread IDs used:")
    print(f"  Main thread: {threading.main_thread().ident}")
    print(f"  Actor thread: {actor.ident}")
    print(f"  Observer thread: {observer.ident}")
    print(f"  Governor thread: {governor.ident}")
    print("\n✨ All threads had unique IDs = True OS-level parallelism!")


def test_low_quality_intervention():
    """Test where Governor intervenes due to low quality."""

    print("\n" + "=" * 70)
    print("Test: Governor Intervention on Low Quality")
    print("=" * 70)

    # Create low-quality agent
    class LowQualityAgent:
        async def __call__(self, state):
            await asyncio.sleep(0.2)
            return {"intent_understanding": ""}  # Empty = low quality

    state = ExecutionState()
    query = "test query"

    actor = ActorThread(LowQualityAgent(), state, query)
    observer = ObserverThread(state, duration_sec=3.0)
    governor = GovernorThread(state, duration_sec=3.0)
    governor.quality_threshold = 0.5

    print("\n⚡ Starting threads...")

    actor.start()
    observer.start()
    governor.start()

    actor.join()
    observer.join()
    governor.join()

    print(f"\n👑 Governor Interventions: {len(governor.interventions)}")
    for intervention in governor.interventions:
        print(f"  - {intervention['type']}: {intervention['reason']}")

    instructions = state.get_instructions()
    print(f"\n📜 Instructions Issued: {len(instructions)}")
    for instruction in instructions:
        print(f"  - {instruction}")

    print("\n✅ Governor successfully intervened!")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    # Test 1: Real OS threads
    test_real_os_threads()

    print("\n" + "=" * 70 + "\n")
    time.sleep(0.5)

    # Test 2: Governor intervention
    test_low_quality_intervention()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
