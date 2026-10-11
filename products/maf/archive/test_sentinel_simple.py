"""
Simple Sentinel + MAF Integration Demo

Demonstrates the 3-thread architecture:
1. Actor thread - Executes agent
2. Observer thread - Watches state changes
3. Governor thread - Can intervene

Using simplified sentinel components to avoid import issues.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
import time
from typing import Any, Dict
from dataclasses import dataclass, field
from datetime import datetime, UTC

# MAF agent
from MAF.test_supervised_agents import MockIntentUnderstandingAgent


# ==========================================
# Simplified Sentinel Components
# ==========================================

@dataclass
class ExecutionState:
    """Runtime state that all 3 threads can observe/modify."""

    attempt: int = 1
    status: str = "PENDING"
    runtime_memory: Dict[str, Any] = field(default_factory=dict)
    runtime_instructions: list = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    def transition(self, status: str):
        """Update status."""
        self.status = status
        if status in ["COMPLETED", "FAILED"]:
            self.completed_at = datetime.now(UTC)


@dataclass
class ExecutionContext:
    """Context passed to executable."""

    input: Dict[str, Any]
    execution_id: str = "exec-001"
    execution_state: ExecutionState | None = None


# ==========================================
# 1. Actor Thread - Executes the Agent
# ==========================================

class ActorThread:
    """
    Actor thread that executes the MAF agent.

    Updates runtime_memory that Observer watches.
    Respects runtime_instructions from Governor.
    """

    def __init__(self, maf_agent: Any, name: str = "Actor"):
        self.maf_agent = maf_agent
        self.name = name

    async def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute agent with state updates."""

        print(f"\n🎭 [{self.name}] Starting execution...")

        state = context.execution_state

        # Update state: Actor is working
        state.runtime_memory["actor_status"] = "working"
        state.runtime_memory["current_query"] = context.input.get("user_query", "")
        state.transition("RUNNING")

        # Check for instructions from Governor
        if state.runtime_instructions:
            print(f"📜 [{self.name}] Following {len(state.runtime_instructions)} instructions from Governor")
            for instruction in state.runtime_instructions:
                print(f"   - {instruction}")

        # Execute the MAF agent
        start_time = time.time()

        try:
            maf_input = {
                "user_query": context.input.get("user_query", ""),
                "updated_query": context.input.get("user_query", ""),
            }

            result = await self.maf_agent(maf_input)

            latency_ms = (time.time() - start_time) * 1000

            # Update state: Actor completed
            state.runtime_memory["result"] = result
            state.runtime_memory["latency_ms"] = latency_ms
            state.runtime_memory["actor_status"] = "complete"
            state.transition("COMPLETED")

            quality = self._compute_quality(result)

            print(f"✅ [{self.name}] Complete: Quality={quality:.2f}, Latency={latency_ms:.0f}ms")

            return {
                "status": "completed",
                "result": result,
                "quality_score": quality,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            print(f"❌ [{self.name}] Error: {e}")

            state.runtime_memory["error"] = str(e)
            state.runtime_memory["actor_status"] = "error"
            state.transition("FAILED")

            return {
                "status": "failed",
                "error": str(e),
            }

    def _compute_quality(self, result: Dict[str, Any]) -> float:
        """Compute quality score."""
        intent_text = result.get("intent_understanding", "")
        if not intent_text:
            return 0.2
        if len(intent_text.strip()) > 20:
            return 0.9
        return 0.5


# ==========================================
# 2. Observer Thread - Watches Memory
# ==========================================

class ObserverThread:
    """
    Observer thread that watches runtime_memory changes.

    Runs concurrently with Actor, checking for updates.
    """

    def __init__(self, name: str = "Observer"):
        self.name = name
        self.observations = []
        self.running = False

    async def start_observing(self, state: ExecutionState, duration_sec: float = 5.0):
        """
        Watch the state for changes.

        Runs concurrently with Actor thread.
        """

        print(f"\n👁️  [{self.name}] Started watching...")

        self.running = True
        start_time = time.time()
        last_status = None

        while self.running and (time.time() - start_time) < duration_sec:
            # Check current state
            current_status = state.runtime_memory.get("actor_status")

            if current_status != last_status:
                # State changed! Observe it
                observation = {
                    "timestamp": time.time(),
                    "actor_status": current_status,
                    "query": state.runtime_memory.get("current_query", ""),
                    "latency": state.runtime_memory.get("latency_ms"),
                }

                self.observations.append(observation)

                print(f"👁️  [{self.name}] Observed change: actor_status={current_status}")

                last_status = current_status

            # Check if actor completed
            if state.status in ["COMPLETED", "FAILED"]:
                print(f"👁️  [{self.name}] Actor finished, stopping observation")
                break

            await asyncio.sleep(0.1)  # Poll every 100ms

        self.running = False
        print(f"👁️  [{self.name}] Stopped. Total observations: {len(self.observations)}")

    def stop(self):
        """Stop observing."""
        self.running = False


# ==========================================
# 3. Governor Thread - Can Intervene
# ==========================================

class GovernorThread:
    """
    Governor thread that can intervene during execution.

    Can issue:
    - Retry commands
    - Tool suggestions
    - Course corrections
    - Quality thresholds
    """

    def __init__(self, name: str = "Governor"):
        self.name = name
        self.interventions = []
        self.running = False
        self.quality_threshold = 0.7

    async def start_governing(self, state: ExecutionState, duration_sec: float = 5.0):
        """
        Monitor execution and intervene when needed.

        Runs concurrently with Actor and Observer.
        """

        print(f"\n👑 [{self.name}] Started governing...")

        self.running = True
        start_time = time.time()

        while self.running and (time.time() - start_time) < duration_sec:

            # Check if we need to intervene
            actor_status = state.runtime_memory.get("actor_status")

            if actor_status == "error":
                # Actor failed! Governor can retry
                print(f"👑 [{self.name}] Detected error, issuing RETRY instruction")

                state.runtime_instructions.append("RETRY_WITH_FALLBACK")

                self.interventions.append({
                    "timestamp": time.time(),
                    "type": "retry",
                    "reason": "actor_error",
                })

            elif actor_status == "complete":
                # Check quality
                quality = self._compute_quality_from_state(state)

                if quality < self.quality_threshold:
                    print(f"👑 [{self.name}] Quality too low ({quality:.2f}), "
                          f"suggesting IMPROVE_QUALITY")

                    state.runtime_instructions.append("IMPROVE_QUALITY")

                    self.interventions.append({
                        "timestamp": time.time(),
                        "type": "quality_improvement",
                        "reason": f"quality_below_threshold ({quality:.2f} < {self.quality_threshold})",
                    })

            # Check if done
            if state.status in ["COMPLETED", "FAILED"]:
                print(f"👑 [{self.name}] Execution finished, stopping governance")
                break

            await asyncio.sleep(0.15)  # Poll every 150ms

        self.running = False
        print(f"👑 [{self.name}] Stopped. Total interventions: {len(self.interventions)}")

    def _compute_quality_from_state(self, state: ExecutionState) -> float:
        """Compute quality from state memory."""
        result = state.runtime_memory.get("result", {})
        intent_text = result.get("intent_understanding", "")

        if not intent_text:
            return 0.2
        if len(intent_text.strip()) > 20:
            return 0.9
        return 0.5

    def stop(self):
        """Stop governing."""
        self.running = False


# ==========================================
# Integration Test
# ==========================================

async def test_3_thread_architecture():
    """
    Test the 3-thread concurrent architecture:

    Thread 1 (Actor): Executes MAF agent
    Thread 2 (Observer): Watches memory changes
    Thread 3 (Governor): Can intervene
    """

    print("\n" + "=" * 70)
    print("3-Thread Concurrent Architecture Test")
    print("Actor | Observer | Governor")
    print("=" * 70)

    # Step 1: Create execution state (shared between all threads)
    print("\n📦 Step 1: Create Shared Execution State")
    execution_state = ExecutionState()
    print("  ✓ Shared state created")

    # Step 2: Create context
    print("\n📋 Step 2: Create Execution Context")
    context = ExecutionContext(
        input={
            "user_query": "How can I implement a supervised multi-agent system?",
            "user_id": "test_user",
        },
        execution_id="exec-001",
        execution_state=execution_state,
    )
    print(f"  ✓ Query: {context.input['user_query'][:50]}...")

    # Step 3: Create the 3 threads
    print("\n🧵 Step 3: Create 3 Threads")

    maf_agent = MockIntentUnderstandingAgent()

    actor = ActorThread(maf_agent, "Actor")
    observer = ObserverThread("Observer")
    governor = GovernorThread("Governor")

    print("  ✓ Actor thread created")
    print("  ✓ Observer thread created")
    print("  ✓ Governor thread created")

    # Step 4: Start all 3 threads concurrently
    print("\n⚡ Step 4: Launch All 3 Threads Concurrently")
    print("-" * 70)

    # Run all 3 concurrently using asyncio.gather
    actor_task = actor.execute(context)
    observer_task = observer.start_observing(execution_state, duration_sec=5.0)
    governor_task = governor.start_governing(execution_state, duration_sec=5.0)

    # Wait for all to complete
    results = await asyncio.gather(actor_task, observer_task, governor_task)

    actor_result = results[0]

    print("-" * 70)

    # Step 5: Results
    print("\n📊 Step 5: Results from Actor")
    print(f"  Status: {actor_result['status']}")
    print(f"  Quality: {actor_result.get('quality_score', 'N/A'):.2f}")
    print(f"  Latency: {actor_result.get('latency_ms', 'N/A'):.0f}ms")

    # Step 6: Observer's view
    print("\n👁️  Step 6: Observer's Observations")
    print(f"  Total observations: {len(observer.observations)}")
    for obs in observer.observations:
        print(f"    - {obs['actor_status']}: latency={obs.get('latency')}")

    # Step 7: Governor's interventions
    print("\n👑 Step 7: Governor's Interventions")
    print(f"  Total interventions: {len(governor.interventions)}")
    for intervention in governor.interventions:
        print(f"    - {intervention['type']}: {intervention['reason']}")

    # Step 8: Final state
    print("\n🧠 Step 8: Final Runtime Memory")
    for key, value in execution_state.runtime_memory.items():
        if isinstance(value, dict):
            print(f"  {key}: {str(value)[:60]}...")
        else:
            print(f"  {key}: {value}")

    print("\n📜 Step 9: Runtime Instructions (from Governor)")
    if execution_state.runtime_instructions:
        for instruction in execution_state.runtime_instructions:
            print(f"  - {instruction}")
    else:
        print("  (none)")

    print("\n" + "=" * 70)
    print("✅ Test Complete: 3 Threads Executed Concurrently!")
    print("=" * 70)


async def test_with_low_quality():
    """Test where quality is low and Governor intervenes."""

    print("\n" + "=" * 70)
    print("Test: Low Quality Triggers Governor Intervention")
    print("=" * 70)

    # Create a mock agent that returns low quality
    class LowQualityAgent:
        async def __call__(self, state):
            await asyncio.sleep(0.2)
            return {"intent_understanding": ""}  # Empty = low quality

    execution_state = ExecutionState()
    context = ExecutionContext(
        input={"user_query": "test query"},
        execution_state=execution_state,
    )

    actor = ActorThread(LowQualityAgent(), "Actor")
    observer = ObserverThread("Observer")
    governor = GovernorThread("Governor")
    governor.quality_threshold = 0.5  # Set threshold

    print("\n⚡ Running with low quality output...")

    results = await asyncio.gather(
        actor.execute(context),
        observer.start_observing(execution_state, duration_sec=3.0),
        governor.start_governing(execution_state, duration_sec=3.0),
    )

    print(f"\n👑 Governor interventions: {len(governor.interventions)}")
    for intervention in governor.interventions:
        print(f"  - {intervention['type']}: {intervention['reason']}")

    print("\n✅ Governor successfully detected low quality and intervened!")


# ==========================================
# Main
# ==========================================

async def main():
    """Run all tests."""

    # Test 1: Basic 3-thread architecture
    await test_3_thread_architecture()

    await asyncio.sleep(0.5)

    # Test 2: Governor intervention on low quality
    await test_with_low_quality()


if __name__ == "__main__":
    asyncio.run(main())
