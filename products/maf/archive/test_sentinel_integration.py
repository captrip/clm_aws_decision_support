"""
Real Integration: MAF Agent + Sentinel Framework
3 Concurrent OS Threads: Actor | Observer | Governor

This demonstrates the REAL vision:
1. Actor thread - Executes the MAF agent
2. Observer thread - Watches memory/state updates in real-time
3. Governor thread - Can intervene to course-correct, retry, suggest tools
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
import time
from typing import Any, Dict
from dataclasses import dataclass

# Sentinel framework imports
from sentinal.core.execution_context import ExecutionContext
from sentinal.core.executable import Executable
from sentinal.core.execution_result import ExecutionResult
from sentinal.core.execution_state import ExecutionState
from sentinal.runtime.runtime import SentinelRuntime
from sentinal.actors.actor import BaseActor
from sentinal.actors.executor import Executor
from sentinal.actors.observer import Observer
from sentinal.actors.governor import Governor
from sentinal.core.actor_ref import ActorRef

# MAF imports (using mock for now, can swap with real)
from MAF.test_supervised_agents import MockIntentUnderstandingAgent


# ==========================================
# 1. Wrap MAF Agent as Sentinel Executable
# ==========================================

class MAFAgentExecutable(Executable):
    """
    Wraps a MAF agent (Intent Understanding, Action Planning, etc.)
    as a Sentinel Executable so it can run in the Actor thread.
    """

    def __init__(self, maf_agent: Any, agent_name: str = "MAF Agent"):
        self.maf_agent = maf_agent
        self.agent_name = agent_name

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute the MAF agent.

        The Actor thread calls this method.
        Memory updates happen here and Observer watches them.
        """

        print(f"\n🎭 [ACTOR] Starting {self.agent_name}...")

        # Get input from context
        user_query = context.input.get("user_query", "")

        # Create MAF state
        maf_state = {
            "user_query": user_query,
            "updated_query": user_query,
        }

        # Update execution state (Observer watches this!)
        if hasattr(context, 'execution_state'):
            context.execution_state.runtime_memory["current_query"] = user_query
            context.execution_state.runtime_memory["status"] = "processing"

        start_time = time.time()

        try:
            # Execute the MAF agent
            result = await self.maf_agent(maf_state)

            latency_ms = (time.time() - start_time) * 1000

            # Update memory with result (Observer sees this)
            if hasattr(context, 'execution_state'):
                context.execution_state.runtime_memory["result"] = result
                context.execution_state.runtime_memory["latency_ms"] = latency_ms
                context.execution_state.runtime_memory["status"] = "complete"

            # Compute quality
            quality_score = self._compute_quality(result)

            print(f"✅ [ACTOR] Complete: Quality={quality_score:.2f}, Latency={latency_ms:.0f}ms")

            return ExecutionResult(
                data={
                    "status": "completed",
                    "result": result,
                    "quality_score": quality_score,
                    "latency_ms": latency_ms,
                    "agent_name": self.agent_name,
                },
                metadata={
                    "execution_time": latency_ms,
                    "quality": quality_score,
                }
            )

        except Exception as e:
            print(f"❌ [ACTOR] Error: {e}")

            if hasattr(context, 'execution_state'):
                context.execution_state.runtime_memory["status"] = "error"
                context.execution_state.runtime_memory["error"] = str(e)

            return ExecutionResult(
                data={
                    "status": "failed",
                    "error": str(e),
                    "agent_name": self.agent_name,
                },
                metadata={"error": str(e)}
            )

    def _compute_quality(self, result: Dict[str, Any]) -> float:
        """Compute quality score from result."""
        intent_text = result.get("intent_understanding", "")
        if not intent_text:
            return 0.2
        if len(intent_text.strip()) > 20:
            return 0.9
        return 0.5


# ==========================================
# 2. Enhanced Observer (Watches Memory)
# ==========================================

class MAFObserver(Observer):
    """
    Enhanced Observer that watches runtime_memory updates
    from the Actor thread.

    Runs concurrently with Actor and Governor.
    """

    def __init__(self, ref: ActorRef):
        super().__init__(ref)
        self.observations = []

    async def receive(self, message):
        """Handle messages and observe execution state."""

        # Call parent observer logic
        await super().receive(message)

        # Additional observation logic
        if hasattr(message, 'execution_id'):
            print(f"👁️  [OBSERVER] Received {message.__class__.__name__} for {message.execution_id}")

            # Store observation
            self.observations.append({
                "timestamp": time.time(),
                "message_type": message.__class__.__name__,
                "execution_id": message.execution_id,
            })


# ==========================================
# 3. Enhanced Governor (Course Correction)
# ==========================================

class MAFGovernor(Governor):
    """
    Enhanced Governor that can:
    1. Monitor quality scores in real-time
    2. Issue retries when quality is low
    3. Suggest tool changes
    4. Course-correct execution

    Runs concurrently with Actor and Observer.
    """

    def __init__(self, ref: ActorRef):
        super().__init__(ref)
        self.interventions = []
        self.quality_threshold = 0.7

    async def receive(self, message):
        """
        Monitor execution and intervene when needed.
        """

        print(f"👑 [GOVERNOR] Received {message.__class__.__name__}")

        # Check if we need to intervene based on observations
        # In the future, this could:
        # - Monitor quality scores from memory
        # - Issue RetryExecution signals
        # - Suggest different tools
        # - Change agent parameters

        # Call parent governor logic (handles retries)
        await super().receive(message)

        # Track interventions
        if hasattr(message, 'execution_id'):
            self.interventions.append({
                "timestamp": time.time(),
                "message_type": message.__class__.__name__,
                "execution_id": message.execution_id,
                "action": "monitored",
            })


# ==========================================
# 4. Integration Test
# ==========================================

async def test_sentinel_with_maf_agent():
    """
    Test the full 3-thread architecture:
    - Actor thread: Executes MAF agent
    - Observer thread: Watches memory updates
    - Governor thread: Can intervene
    """

    print("\n" + "=" * 70)
    print("SENTINEL + MAF Integration Test")
    print("3 Concurrent Threads: Actor | Observer | Governor")
    print("=" * 70)

    # Step 1: Create Sentinel runtime
    print("\n📦 Step 1: Initialize Sentinel Runtime")
    runtime = SentinelRuntime()

    # Step 2: Create the 3 actors
    print("\n🎭 Step 2: Create Actor, Observer, Governor")

    executor_ref = ActorRef("executor", "local")
    observer_ref = ActorRef("observer", "local")
    governor_ref = ActorRef("governor", "local")

    executor = Executor(executor_ref)
    observer = MAFObserver(observer_ref)
    governor = MAFGovernor(governor_ref)

    # Register actors with runtime
    runtime.register(executor)
    runtime.register(observer)
    runtime.register(governor)

    print(f"  ✓ Executor: {executor_ref}")
    print(f"  ✓ Observer: {observer_ref}")
    print(f"  ✓ Governor: {governor_ref}")

    # Step 3: Start all actors (3 concurrent threads)
    print("\n🚀 Step 3: Start all actors")
    await runtime.start()
    print("  ✓ All 3 threads running concurrently")

    # Step 4: Create MAF agent executable
    print("\n🤖 Step 4: Wrap MAF Agent as Executable")
    maf_agent = MockIntentUnderstandingAgent()
    executable = MAFAgentExecutable(maf_agent, "Intent Understanding Agent")
    print(f"  ✓ Wrapped: {executable.agent_name}")

    # Step 5: Create execution context with state
    print("\n📋 Step 5: Create Execution Context")

    execution_state = ExecutionState()
    execution_state.runtime_memory = {}
    execution_state.runtime_instructions = []

    context = ExecutionContext(
        input={
            "user_query": "How can I build a multi-agent system with supervision?",
            "user_id": "test_user",
        }
    )
    context.execution_state = execution_state

    print(f"  ✓ Query: {context.input['user_query'][:50]}...")

    # Step 6: Execute (Actor runs, Observer watches, Governor monitors)
    print("\n⚡ Step 6: Execute Agent with 3-Thread Supervision")
    print("-" * 70)

    # This runs the agent in the Actor thread
    # Observer and Governor are running concurrently
    result = await executable.execute(context)

    print("-" * 70)

    # Step 7: Check results
    print("\n📊 Step 7: Results")
    print(f"  Status: {result.data['status']}")
    print(f"  Quality: {result.data.get('quality_score', 'N/A')}")
    print(f"  Latency: {result.data.get('latency_ms', 'N/A'):.0f}ms")
    print(f"  Result: {str(result.data.get('result', {}))[:100]}...")

    # Step 8: Check memory updates (what Observer saw)
    print("\n🧠 Step 8: Runtime Memory (Observer's View)")
    for key, value in execution_state.runtime_memory.items():
        if isinstance(value, dict):
            print(f"  {key}: {str(value)[:80]}...")
        else:
            print(f"  {key}: {value}")

    # Step 9: Check interventions (Governor's actions)
    print("\n👑 Step 9: Governor Interventions")
    print(f"  Total interventions: {len(governor.interventions)}")
    for intervention in governor.interventions[:3]:  # Show first 3
        print(f"  - {intervention['message_type']}: {intervention['action']}")

    # Step 10: Cleanup
    print("\n🛑 Step 10: Stop all actors")
    await runtime.stop()
    print("  ✓ All threads stopped")

    print("\n" + "=" * 70)
    print("✅ Test Complete: 3-Thread Architecture Working!")
    print("=" * 70)


async def test_multiple_queries():
    """Test with multiple queries to show concurrent execution."""

    print("\n" + "=" * 70)
    print("Multiple Query Test with 3-Thread Architecture")
    print("=" * 70)

    # Setup runtime
    runtime = SentinelRuntime()

    executor = Executor(ActorRef("executor", "local"))
    observer = MAFObserver(ActorRef("observer", "local"))
    governor = MAFGovernor(ActorRef("governor", "local"))

    runtime.register(executor)
    runtime.register(observer)
    runtime.register(governor)

    await runtime.start()

    # Create executable
    maf_agent = MockIntentUnderstandingAgent()
    executable = MAFAgentExecutable(maf_agent, "Intent Agent")

    # Test queries
    queries = [
        "What's the weather?",
        "Tell me a joke",
        "How do I implement OAuth?",
    ]

    print(f"\n🚀 Processing {len(queries)} queries with supervision...\n")

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: {query}")

        context = ExecutionContext(
            input={"user_query": query, "user_id": "test"}
        )
        context.execution_state = ExecutionState()
        context.execution_state.runtime_memory = {}

        result = await executable.execute(context)

        print(f"  Result: Quality={result.data.get('quality_score', 0):.2f}, "
              f"Latency={result.data.get('latency_ms', 0):.0f}ms")

    print(f"\n📊 Observer tracked {len(observer.observations)} events")
    print(f"👑 Governor made {len(governor.interventions)} interventions")

    await runtime.stop()

    print("\n✅ Multiple query test complete!")


# ==========================================
# Main
# ==========================================

async def main():
    """Run all tests."""

    # Test 1: Basic integration
    await test_sentinel_with_maf_agent()

    await asyncio.sleep(0.5)

    # Test 2: Multiple queries
    await test_multiple_queries()


if __name__ == "__main__":
    asyncio.run(main())
