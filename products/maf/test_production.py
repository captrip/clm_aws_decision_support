"""
Production Test - Sentinel-Integrated MAF Orchestrator

Simple test to verify the production orchestrator works.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio


async def test_with_mock_agents():
    """Test with mock agents (no LLM dependencies)."""

    print("\n" + "=" * 70)
    print("PRODUCTION TEST - Mock Agents")
    print("=" * 70)

    from MAF.global_.sentinel_orchestrator import SentinelOrchestrator

    # Create mock agent
    class MockIntentAgent:
        name = "Intent Understanding Agent"

        async def __call__(self, state):
            await asyncio.sleep(0.2)
            return {
                "intent_understanding": f"Intent: {state.get('user_query', '')}"
            }

    # Create orchestrator with supervision enabled
    print("\n📦 Creating Sentinel Orchestrator...")
    orc = SentinelOrchestrator(
        quality_threshold=0.75,
        enable_supervision=True,
    )

    # Add mock agent
    orc.add_supervised_agent("Intent Understanding", MockIntentAgent())

    print("\n⚡ Executing with supervision...")

    state = {
        "user_id": "test_123",
        "user_query": "How do I build a secure authentication system?",
    }

    result = await orc.ainvoke(state)

    print("\n📊 Results:")
    print(f"  Query: {result['user_query']}")
    print(f"  Intent: {result['intent_understanding'].get('intent_understanding', 'N/A')[:60]}...")
    print(f"  Supervision: {result['supervision_enabled']}")

    if result['supervision_metadata']:
        meta = result['supervision_metadata']
        print(f"\n👁️  Observer Observations: {len(meta['observer_observations'])}")
        print(f"👑 Governor Interventions: {len(meta['governor_interventions'])}")

    print("\n📈 Orchestrator Stats:")
    stats = result['orchestrator_stats']
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Test Complete!")


async def test_with_real_agents():
    """Test with real MAF agents (requires dependencies)."""

    print("\n" + "=" * 70)
    print("PRODUCTION TEST - Real MAF Agents")
    print("=" * 70)

    try:
        from MAF.global_.sentinel_orchestrator import SentinelOrchestrator

        print("\n📦 Creating Sentinel Orchestrator with real agents...")
        orc = SentinelOrchestrator(
            auto_load_skills=True,
            db_path="subgraph_registry.db",
            quality_threshold=0.75,
            enable_supervision=True,
        ).build()

        print("\n⚡ Executing with real LLM agents...")

        state = {
            "user_id": "test_123",
            "user_query": "How do I implement OAuth authentication?",
        }

        result = await orc.ainvoke(state)

        print("\n📊 Results:")
        print(f"  Query: {result['user_query']}")

        intent = result['intent_understanding'].get('intent_understanding', '')
        print(f"\n📤 Intent Understanding (from LLM):")
        print("─" * 70)
        print(intent[:300] if intent else "(empty)")
        if len(intent) > 300:
            print(f"... ({len(intent) - 300} more chars)")
        print("─" * 70)

        if result['supervision_metadata']:
            meta = result['supervision_metadata']
            print(f"\n👁️  Observer Observations: {len(meta['observer_observations'])}")
            print(f"👑 Governor Interventions: {len(meta['governor_interventions'])}")

        print("\n✅ Test Complete with Real Agents!")

    except ImportError as e:
        print(f"\n⚠️  Could not load real agents: {e}")
        print("   Run test_with_mock_agents() instead")


async def main():
    """Run tests."""

    print("\n" + "=" * 70)
    print("🚀 PRODUCTION ORCHESTRATOR TESTS")
    print("=" * 70)

    # Test 1: Mock agents (always works)
    await test_with_mock_agents()

    await asyncio.sleep(0.5)

    # Test 2: Real agents (requires dependencies)
    print("\n" + "=" * 70 + "\n")
    await test_with_real_agents()


if __name__ == "__main__":
    asyncio.run(main())
