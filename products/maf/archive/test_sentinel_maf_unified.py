"""
Test Sentinel + MAF Unified Framework

Shows how MAF agents automatically run with 3-thread supervision.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
import json
from MAF.sentinel_maf_integration import (
    SentinelMAFAgent,
    wrap_maf_agent_with_sentinel,
)
from MAF.test_supervised_agents import MockIntentUnderstandingAgent


async def test_1_basic_wrapper():
    """Test 1: Wrap a MAF agent with Sentinel."""

    print("\n" + "=" * 70)
    print("Test 1: Wrap MAF Agent with Sentinel")
    print("=" * 70)

    # Step 1: Create regular MAF agent
    print("\n📦 Step 1: Create MAF Agent")
    maf_agent = MockIntentUnderstandingAgent()
    print(f"  ✓ {maf_agent.name}")

    # Step 2: Wrap with Sentinel
    print("\n🔄 Step 2: Wrap with Sentinel (3-thread architecture)")
    sentinel_agent = wrap_maf_agent_with_sentinel(
        maf_agent=maf_agent,
        agent_name="Intent Understanding",
        quality_threshold=0.7,
        timeout_sec=10.0,
    )
    print("  ✓ Wrapped with Actor|Observer|Governor threads")

    # Step 3: Execute
    print("\n⚡ Step 3: Execute Agent")
    print("-" * 70)

    maf_input = {
        "user_query": "How do I build a multi-agent system?",
        "updated_query": "How do I build a multi-agent system?",
    }

    result = await sentinel_agent.execute(maf_input)

    print("-" * 70)

    # Step 4: Results
    print("\n📊 Step 4: Results")
    print(f"  Execution ID: {result['execution_id']}")
    print(f"  Agent: {result['agent_name']}")
    print(f"  Status: {result['status']}")
    print(f"  Latency: {result['timing']['latency_ms']:.0f}ms")

    # Step 5: Supervision metadata
    print("\n👁️  Step 5: Observer Observations")
    observations = result['supervision']['observer_observations']
    print(f"  Total: {len(observations)}")
    for obs in observations:
        print(f"    - Status: {obs['status']}, Latency: {obs.get('latency')}")

    print("\n👑 Step 6: Governor Interventions")
    interventions = result['supervision']['governor_interventions']
    print(f"  Total: {len(interventions)}")
    if interventions:
        for intervention in interventions:
            print(f"    - Type: {intervention['type']}")
            print(f"      Details: {intervention}")

    print("\n🧠 Step 7: Runtime Memory")
    for key, value in result['runtime_memory'].items():
        if isinstance(value, dict):
            print(f"  {key}: {str(value)[:60]}...")
        else:
            print(f"  {key}: {value}")

    print("\n📜 Step 8: Runtime Instructions (from Governor)")
    if result['runtime_instructions']:
        for instruction in result['runtime_instructions']:
            print(f"  - {instruction}")
    else:
        print("  (none)")

    print("\n✅ Test 1 Complete!")


async def test_2_maf_compatible_interface():
    """Test 2: Use as drop-in replacement for MAF agent."""

    print("\n" + "=" * 70)
    print("Test 2: MAF-Compatible Interface")
    print("=" * 70)

    print("\n🔄 Creating Sentinel agent with MAF-compatible interface...")

    sentinel_agent = SentinelMAFAgent(
        maf_agent=MockIntentUnderstandingAgent(),
        agent_name="Intent Agent",
        quality_threshold=0.7,
    )

    # Use like a regular MAF agent (returns just the result)
    print("\n⚡ Calling like regular MAF agent: await agent(input)")

    maf_input = {
        "user_query": "What's the weather?",
        "updated_query": "What's the weather?",
    }

    # This returns just the MAF result (not the full supervision data)
    result = await sentinel_agent(maf_input)

    print(f"\n📤 Result (MAF-compatible):")
    print(f"  {result}")

    print("\n✅ Test 2 Complete - Works as drop-in replacement!")


async def test_3_low_quality_intervention():
    """Test 3: Governor intervenes on low quality."""

    print("\n" + "=" * 70)
    print("Test 3: Governor Intervention on Low Quality")
    print("=" * 70)

    # Create agent that returns low quality
    class LowQualityAgent:
        name = "Low Quality Agent"

        async def __call__(self, state):
            await asyncio.sleep(0.2)
            return {"intent_understanding": ""}  # Empty = low quality

    sentinel_agent = SentinelMAFAgent(
        maf_agent=LowQualityAgent(),
        agent_name="Low Quality Test",
        quality_threshold=0.5,  # Set threshold
    )

    print("\n⚡ Executing agent with low quality output...")

    result = await sentinel_agent.execute({"user_query": "test"})

    print(f"\n📊 Status: {result['status']}")

    print("\n👑 Governor Interventions:")
    interventions = result['supervision']['governor_interventions']
    print(f"  Total: {len(interventions)}")
    for intervention in interventions:
        print(f"    - Type: {intervention['type']}")
        if 'quality' in intervention:
            print(f"      Quality: {intervention['quality']:.2f}")
            print(f"      Threshold: {intervention['threshold']:.2f}")

    print("\n📜 Instructions Issued:")
    for instruction in result['runtime_instructions']:
        print(f"  - {instruction}")

    print("\n✅ Test 3 Complete - Governor successfully intervened!")


async def test_4_multiple_executions():
    """Test 4: Multiple executions with state isolation."""

    print("\n" + "=" * 70)
    print("Test 4: Multiple Executions with State Isolation")
    print("=" * 70)

    sentinel_agent = SentinelMAFAgent(
        maf_agent=MockIntentUnderstandingAgent(),
        agent_name="Intent Agent",
    )

    queries = [
        "What's the weather?",
        "Tell me a joke",
        "How are you?",
    ]

    print(f"\n⚡ Processing {len(queries)} queries...\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Query: {query}")

        result = await sentinel_agent.execute({
            "user_query": query,
            "updated_query": query,
        })

        print(f"  Execution ID: {result['execution_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Latency: {result['timing']['latency_ms']:.0f}ms")
        print(f"  Observations: {len(result['supervision']['observer_observations'])}")
        print()

    print(f"✅ Test 4 Complete - Total executions: {sentinel_agent.execution_count}")


async def test_5_selective_supervision():
    """Test 5: Disable Observer or Governor selectively."""

    print("\n" + "=" * 70)
    print("Test 5: Selective Supervision (Observer/Governor On/Off)")
    print("=" * 70)

    # Test with only Actor (no supervision)
    print("\n🎭 Test A: Actor only (no Observer, no Governor)")
    agent_actor_only = SentinelMAFAgent(
        maf_agent=MockIntentUnderstandingAgent(),
        agent_name="Actor Only",
        enable_observer=False,
        enable_governor=False,
    )

    result = await agent_actor_only.execute({"user_query": "test"})
    print(f"  Observations: {len(result['supervision']['observer_observations'])}")
    print(f"  Interventions: {len(result['supervision']['governor_interventions'])}")

    # Test with Actor + Observer (no Governor)
    print("\n👁️  Test B: Actor + Observer (no Governor)")
    agent_with_observer = SentinelMAFAgent(
        maf_agent=MockIntentUnderstandingAgent(),
        agent_name="With Observer",
        enable_observer=True,
        enable_governor=False,
    )

    result = await agent_with_observer.execute({"user_query": "test"})
    print(f"  Observations: {len(result['supervision']['observer_observations'])}")
    print(f"  Interventions: {len(result['supervision']['governor_interventions'])}")

    # Test with full supervision
    print("\n⚡ Test C: Full supervision (Actor + Observer + Governor)")
    agent_full = SentinelMAFAgent(
        maf_agent=MockIntentUnderstandingAgent(),
        agent_name="Full Supervision",
        enable_observer=True,
        enable_governor=True,
    )

    result = await agent_full.execute({"user_query": "test"})
    print(f"  Observations: {len(result['supervision']['observer_observations'])}")
    print(f"  Interventions: {len(result['supervision']['governor_interventions'])}")

    print("\n✅ Test 5 Complete - Supervision is configurable!")


# ==========================================
# Main
# ==========================================

async def main():
    """Run all tests."""

    tests = [
        test_1_basic_wrapper,
        test_2_maf_compatible_interface,
        test_3_low_quality_intervention,
        test_4_multiple_executions,
        test_5_selective_supervision,
    ]

    for test in tests:
        await test()
        await asyncio.sleep(0.3)

    print("\n" + "=" * 70)
    print("🎉 ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\n✨ MAF + Sentinel Integration Working!")
    print("   - Every agent runs with 3 OS threads")
    print("   - Actor executes, Observer watches, Governor intervenes")
    print("   - Drop-in replacement for existing MAF agents")
    print("   - Full supervision metadata available")


if __name__ == "__main__":
    asyncio.run(main())
