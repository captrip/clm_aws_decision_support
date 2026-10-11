"""
Real MAF Supervised Agents Integration Test

Tests the supervised agent wrapper with REAL MAF agents using LLMs.
This uses the actual IntentUnderstandingAgent with LangGraph and LLMs.
"""

import sys
from pathlib import Path

# Add workspace root to path for imports
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
import time
from typing import Any, Dict

from experiment.agent_supervision import (
    SupervisedAgentState,
    AgentStatus,
)
from MAF.supervised_subagents import SubagentObserver, SubagentGovernor

# Import REAL MAF agents
from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent
from MAF.global_.state import Orchestrator_Global_State


# Supervised wrapper for REAL IntentUnderstandingAgent
class SupervisedIntentUnderstandingAgent:
    """Wraps the REAL MAF IntentUnderstandingAgent with full supervision."""

    def __init__(self, original_agent: Any = None):
        if original_agent is None:
            # Use the REAL MAF agent
            self.original_agent = IntentUnderstandingAgent()
        else:
            self.original_agent = original_agent

        self.name = f"Supervised-{self.original_agent.name}"
        self.description = f"Supervised {self.original_agent.description}"

        # Supervision state
        self.state = SupervisedAgentState()
        self.observer = None
        self.governor = None

        # Metrics
        self.execution_count = 0
        self.intervention_count = 0

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with supervision."""
        self.execution_count += 1

        self.state.update("status", AgentStatus.WORKING)
        self.state.update("current_operation", "Intent Understanding")

        # Start observer and governor
        self.observer = SubagentObserver(self.state, "intent")
        self.governor = SubagentGovernor(self.state, "intent")

        self.observer.start()
        self.governor.start()

        try:
            # Execute original REAL agent
            start_time = time.time()
            result = await self.original_agent(state)
            latency_ms = (time.time() - start_time) * 1000

            # Compute quality
            quality_score = self._compute_quality(result)

            # Update metrics
            self.state.add_metric("latency_ms", latency_ms)
            self.state.add_metric("quality_score", quality_score)

            self.state.update("status", AgentStatus.COMPLETE)

            print(
                f"✅ [Real Intent Agent] Quality: {quality_score:.2f} | Latency: {latency_ms:.0f}ms"
            )

            return result

        except Exception as e:
            self.state.update("status", AgentStatus.ERROR)
            self.state.add_error(str(e))
            print(f"❌ [Real Intent Agent] Error: {e}")
            raise

        finally:
            # Stop monitoring
            if self.observer:
                self.observer.stop()
            if self.governor:
                self.governor.stop()
            if self.observer:
                self.observer.join(timeout=1)
            if self.governor:
                self.governor.join(timeout=1)

    def _compute_quality(self, result: Dict[str, Any]) -> float:
        """Compute quality score based on output richness."""
        intent_text = result.get("intent_understanding", "")

        if not intent_text:
            return 0.1

        intent_len = len(intent_text.strip())

        # Score based on completeness
        if intent_len == 0:
            return 0.1
        elif intent_len < 50:
            return 0.4
        elif intent_len < 150:
            return 0.7
        else:
            # Rich output with both goals and risks
            score = 0.8
            if "GOALS" in intent_text and "RISK" in intent_text:
                score = 0.95
            return score


# ==========================================
# Integration Tests with REAL Agents
# ==========================================


async def test_1_real_basic_supervision():
    """Test 1: Basic supervised agent execution with REAL LLM."""
    print("\n" + "=" * 70)
    print("TEST 1: Real Agent - Basic Supervised Execution")
    print("=" * 70)

    # Create supervised agent wrapping REAL agent
    supervised = SupervisedIntentUnderstandingAgent()

    print(f"\nAgent name: {supervised.name}")
    print(f"Agent description: {supervised.description}")

    # Execute with REAL state
    print("\n🚀 Executing with REAL LLM...")

    state = Orchestrator_Global_State(
        user_id="test_user",
        user_query="What's the weather like today?",
        updated_query="What's the weather like today?",
    )

    result = await supervised(state)

    print(f"\n📤 Result:")
    print(f"  Intent Understanding: {result.get('intent_understanding', 'N/A')[:200]}...")
    print(f"\n📊 Stats: executions={supervised.execution_count}")

    assert supervised.execution_count == 1
    assert "intent_understanding" in result
    print("✅ PASS")


async def test_2_real_multiple_queries():
    """Test 2: Multiple real queries."""
    print("\n" + "=" * 70)
    print("TEST 2: Real Agent - Multiple Queries")
    print("=" * 70)

    supervised = SupervisedIntentUnderstandingAgent()

    queries = [
        "Tell me a joke about AI",
        "How can I build a multi-agent system?",
        "What are the best practices for prompt engineering?",
    ]

    print(f"\n🚀 Processing {len(queries)} real queries with LLM...\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Query: {query}")

        state = Orchestrator_Global_State(
            user_id="test_user",
            user_query=query,
            updated_query=query,
        )

        result = await supervised(state)

        intent = result.get("intent_understanding", "")
        print(f"  Intent (preview): {intent[:100]}...")
        print()

    print(f"✅ All {supervised.execution_count} executions complete")
    assert supervised.execution_count == len(queries)
    print("✅ PASS")


async def test_3_real_quality_metrics():
    """Test 3: Quality scoring with real outputs."""
    print("\n" + "=" * 70)
    print("TEST 3: Real Agent - Quality Metrics")
    print("=" * 70)

    supervised = SupervisedIntentUnderstandingAgent()

    query = "I need help building a secure authentication system for my web app"

    print(f"\n🚀 Processing complex query: '{query}'")

    state = Orchestrator_Global_State(
        user_id="test_user",
        user_query=query,
        updated_query=query,
    )

    result = await supervised(state)

    print(f"\n📤 Full Intent Understanding Output:")
    print("=" * 70)
    print(result.get("intent_understanding", "N/A"))
    print("=" * 70)

    # Check metrics
    metrics = supervised.state.get("metrics", {})
    print(f"\n📊 Captured Metrics:")
    for key, values in metrics.items():
        if isinstance(values, list) and values:
            avg = sum(values) / len(values)
            print(f"  {key}: {avg:.2f}")
        else:
            print(f"  {key}: {values}")

    print("✅ PASS")


async def test_4_real_integration_with_orchestrator():
    """Test 4: Show how to integrate with real orchestrator."""
    print("\n" + "=" * 70)
    print("TEST 4: Real Agent - Orchestrator Integration Pattern")
    print("=" * 70)

    print("\nOriginal way (without supervision):")
    print("-" * 70)

    # Original approach
    original_agent = IntentUnderstandingAgent()
    print(f"Agent: {original_agent.name}")

    print("\nSupervised way (with full monitoring):")
    print("-" * 70)

    # Supervised approach
    supervised_agent = SupervisedIntentUnderstandingAgent(original_agent)
    print(f"Agent: {supervised_agent.name}")

    print("\n🚀 Execute both for comparison...")

    state = Orchestrator_Global_State(
        user_id="test_user",
        user_query="Help me optimize my database queries",
        updated_query="Help me optimize my database queries",
    )

    # Original (no supervision)
    print("\n[Original Agent]")
    start = time.time()
    result1 = await original_agent(state)
    elapsed1 = (time.time() - start) * 1000
    print(f"  Latency: {elapsed1:.0f}ms")
    print(f"  Output length: {len(result1.get('intent_understanding', ''))}")

    # Supervised (with monitoring)
    print("\n[Supervised Agent]")
    result2 = await supervised_agent(state)
    print(f"  Output length: {len(result2.get('intent_understanding', ''))}")
    print(f"  Intervention count: {supervised_agent.intervention_count}")

    print("\n✅ Both approaches work - supervised adds monitoring!")
    print("✅ PASS")


async def test_5_real_custom_query():
    """Test 5: Interactive test with custom query."""
    print("\n" + "=" * 70)
    print("TEST 5: Real Agent - Custom User Query")
    print("=" * 70)

    # Get custom query from command line or use default
    custom_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Can you help me design a multi-agent system with supervision and monitoring?"

    print(f"\n💬 Custom Query: {custom_query}")

    supervised = SupervisedIntentUnderstandingAgent()

    state = Orchestrator_Global_State(
        user_id="custom_user",
        user_query=custom_query,
        updated_query=custom_query,
    )

    print(f"\n🚀 Processing with REAL supervised agent...")
    result = await supervised(state)

    print(f"\n{'=' * 70}")
    print("FULL INTENT UNDERSTANDING OUTPUT")
    print('=' * 70)
    print(result.get("intent_understanding", "N/A"))
    print('=' * 70)

    # Show detailed metrics
    print(f"\n📊 Detailed Supervision Metrics:")
    print(f"  Status: {supervised.state.get('status')}")
    print(f"  Executions: {supervised.execution_count}")
    print(f"  Interventions: {supervised.intervention_count}")

    metrics = supervised.state.get("metrics", {})
    for key, values in metrics.items():
        if isinstance(values, list) and values:
            avg = sum(values) / len(values)
            print(f"  {key}: {avg:.2f}")

    print("✅ PASS")


# ==========================================
# Run all tests
# ==========================================


async def main():
    print("\n" + "=" * 70)
    print("REAL MAF Supervised Agents Integration Tests")
    print("Using ACTUAL IntentUnderstandingAgent with LLMs")
    print("=" * 70)

    tests = [
        test_1_real_basic_supervision,
        test_2_real_multiple_queries,
        test_3_real_quality_metrics,
        test_4_real_integration_with_orchestrator,
        test_5_real_custom_query,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAIL: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("✅ All tests passed with REAL agents!")
    else:
        print(f"❌ {failed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
