"""
MAF Supervised Agents Integration Test

Tests the supervised agent wrapper without requiring full MAF dependencies.
Demonstrates how the wrapper integrates with MAF agents.
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
    CorrectionType,
)
from MAF.supervised_subagents import SubagentObserver, SubagentGovernor


# Mock MAF Agent (simulates IntentUnderstandingAgent without dependencies)
class MockIntentUnderstandingAgent:
    """Mock MAF agent for testing."""
    
    def __init__(self, name: str = "Intent Understanding Agent"):
        self.description = "Intent Understanding"
        self.name = name
        self.call_count = 0
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent execution."""
        self.call_count += 1
        await asyncio.sleep(0.1)  # Simulate work
        
        user_query = state.get("user_query", "")
        
        return {
            "intent_understanding": f"Intent: {user_query[:50] if user_query else 'unknown'}"
        }


# Supervised wrapper (production version)
class SupervisedIntentUnderstandingAgent:
    """Wraps MAF IntentUnderstandingAgent with full supervision."""
    
    def __init__(self, original_agent: Any = None):
        if original_agent is None:
            self.original_agent = MockIntentUnderstandingAgent()
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
            # Execute original agent
            start_time = time.time()
            result = await self.original_agent(state)
            latency_ms = (time.time() - start_time) * 1000
            
            # Compute quality
            quality_score = self._compute_quality(result)
            
            # Update metrics
            self.state.add_metric("latency_ms", latency_ms)
            self.state.add_metric("quality_score", quality_score)
            
            self.state.update("status", AgentStatus.COMPLETE)
            
            print(f"✅ [Intent Agent] Quality: {quality_score:.2f} | Latency: {latency_ms:.0f}ms")
            
            return result
        
        except Exception as e:
            self.state.update("status", AgentStatus.ERROR)
            self.state.add_error(str(e))
            print(f"❌ [Intent Agent] Error: {e}")
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
        """Compute quality score."""
        intent_text = result.get("intent_understanding", "")
        if not intent_text:
            return 0.2
        if isinstance(intent_text, str) and len(intent_text.strip()) > 0:
            return 0.9
        return 0.5


# ==========================================
# Integration Tests
# ==========================================

async def test_1_basic_supervision():
    """Test 1: Basic supervised agent execution."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Supervised Agent Execution")
    print("=" * 70)
    
    # Create supervised agent wrapping mock agent
    mock_agent = MockIntentUnderstandingAgent()
    supervised = SupervisedIntentUnderstandingAgent(mock_agent)
    
    print(f"\nAgent name: {supervised.name}")
    print(f"Agent description: {supervised.description}")
    
    # Execute
    print("\n🚀 Executing...")
    result = await supervised({
        "user_query": "What's the weather?",
        "updated_query": "What's the weather?",
    })
    
    print(f"\n📤 Result: {result}")
    print(f"📊 Stats: executions={supervised.execution_count}, interventions={supervised.intervention_count}")
    
    assert supervised.execution_count == 1
    assert "intent_understanding" in result
    print("✅ PASS")


async def test_2_multiple_executions():
    """Test 2: Multiple executions accumulate metrics."""
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Executions")
    print("=" * 70)
    
    supervised = SupervisedIntentUnderstandingAgent()
    
    queries = [
        "Tell me a joke",
        "What's today's weather?",
        "How are you?",
    ]
    
    print(f"\n🚀 Executing {len(queries)} queries...")
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n  [{i}/{len(queries)}] {query}")
        result = await supervised({
            "user_query": query,
            "updated_query": query,
        })
        results.append(result)
    
    print(f"\n✅ All {len(results)} executions complete")
    print(f"📊 Agent stats: executions={supervised.execution_count}")
    
    assert supervised.execution_count == len(queries)
    assert len(results) == len(queries)
    print("✅ PASS")


async def test_3_error_handling():
    """Test 3: Error handling and recovery."""
    print("\n" + "=" * 70)
    print("TEST 3: Error Handling")
    print("=" * 70)
    
    class FailingAgent:
        def __init__(self):
            self.name = "Failing Agent"
            self.description = "Always fails"
            self.attempt = 0
        
        async def __call__(self, state):
            self.attempt += 1
            await asyncio.sleep(0.05)
            if self.attempt < 2:
                raise ValueError("Temporary failure")
            return {"intent_understanding": "Recovered!"}
    
    supervised = SupervisedIntentUnderstandingAgent(FailingAgent())
    
    print("\n🚀 First execution (will fail)...")
    try:
        result = await supervised({"user_query": "test"})
        print("❌ Should have raised error")
        assert False
    except ValueError as e:
        print(f"✓ Error caught as expected: {e}")
    
    print("\n🚀 Second execution (should recover)...")
    result = await supervised({"user_query": "test"})
    print(f"✓ Recovery successful: {result}")
    
    print("✅ PASS")


async def test_4_integration_pattern():
    """Test 4: Integration pattern with orchestrator."""
    print("\n" + "=" * 70)
    print("TEST 4: Integration Pattern (How to use in Orchestrator)")
    print("=" * 70)
    
    print("\nPattern 1: Direct replacement")
    print("-" * 70)
    
    # Original agent
    original = MockIntentUnderstandingAgent("Original Intent Agent")
    
    # Wrap with supervision
    supervised = SupervisedIntentUnderstandingAgent(original)
    
    print(f"Original agent: {original.name}")
    print(f"Wrapped agent:  {supervised.name}")
    
    # Use in orchestrator
    print("\n🚀 Using in orchestrator...")
    
    orchestrator_state = {
        "user_query": "What's the meaning of life?",
        "updated_query": "What's the meaning of life?",
        "user_id": "user_123",
    }
    
    result = await supervised(orchestrator_state)
    print(f"\n✓ Orchestrator call successful")
    print(f"  Result: {result}")
    
    print("\nPattern 2: Factory function")
    print("-" * 70)
    
    def create_supervised_intent_agent():
        """Factory for creating supervised intent agent."""
        return SupervisedIntentUnderstandingAgent()
    
    supervised2 = create_supervised_intent_agent()
    result2 = await supervised2(orchestrator_state)
    print(f"✓ Factory created agent works")
    
    print("✅ PASS")


async def test_5_state_isolation():
    """Test 5: Multiple agent instances have isolated state."""
    print("\n" + "=" * 70)
    print("TEST 5: State Isolation Between Agents")
    print("=" * 70)
    
    agent1 = SupervisedIntentUnderstandingAgent()
    agent2 = SupervisedIntentUnderstandingAgent()
    
    print(f"\nAgent 1 state id: {id(agent1.state)}")
    print(f"Agent 2 state id: {id(agent2.state)}")
    print(f"States are isolated: {agent1.state is not agent2.state}")
    
    print("\n🚀 Execute both agents...")
    r1 = await agent1({"user_query": "Agent 1 query"})
    r2 = await agent2({"user_query": "Agent 2 query"})
    
    print(f"\nAgent 1: executions={agent1.execution_count}")
    print(f"Agent 2: executions={agent2.execution_count}")
    
    assert agent1.execution_count == 1
    assert agent2.execution_count == 1
    assert agent1.state is not agent2.state
    
    print("✅ PASS")


# ==========================================
# Run all tests
# ==========================================

async def main():
    print("\n" + "=" * 70)
    print("MAF Supervised Agents Integration Tests")
    print("=" * 70)
    
    tests = [
        test_1_basic_supervision,
        test_2_multiple_executions,
        test_3_error_handling,
        test_4_integration_pattern,
        test_5_state_isolation,
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
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
