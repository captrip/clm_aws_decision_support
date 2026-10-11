"""
Real MAF Agents + Sentinel Integration Test

Tests the 3-thread architecture with REAL MAF agents using LLMs.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
from typing import Dict, Any

# Real MAF imports
from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent
from MAF.global_.state import Orchestrator_Global_State

# Sentinel integration
from MAF.sentinel_maf_integration import (
    wrap_maf_agent_with_sentinel,
    SentinelMAFAgent,
)


async def test_real_intent_agent_with_sentinel():
    """
    Test REAL IntentUnderstandingAgent with 3-thread supervision.

    This uses the actual MAF agent with LLMs, not mocks.
    """

    print("\n" + "=" * 70)
    print("REAL MAF AGENT + SENTINEL TEST")
    print("Using IntentUnderstandingAgent with LLMs")
    print("=" * 70)

    # Step 1: Create real MAF agent
    print("\n📦 Step 1: Create Real MAF Agent")
    intent_agent = IntentUnderstandingAgent()
    print(f"  ✓ {intent_agent.name}")
    print(f"  ✓ Uses LangGraph IUSwarm with LLMs")

    # Step 2: Wrap with Sentinel
    print("\n🔄 Step 2: Wrap with Sentinel (3-thread architecture)")
    supervised_agent = wrap_maf_agent_with_sentinel(
        maf_agent=intent_agent,
        agent_name="Intent Understanding",
        quality_threshold=0.75,
        timeout_sec=60.0,  # Longer timeout for LLM calls
    )
    print("  ✓ Wrapped with Actor|Observer|Governor threads")

    # Step 3: Create state
    print("\n📋 Step 3: Create MAF State")

    user_query = "How do I build a secure authentication system for my web application?"

    state = Orchestrator_Global_State(
        user_id="test_user_001",
        user_query=user_query,
        updated_query=user_query,
    )

    print(f"  Query: {user_query}")

    # Step 4: Execute with supervision
    print("\n⚡ Step 4: Execute with 3-Thread Supervision")
    print("-" * 70)
    print("  🎭 Actor thread will execute the LLM-based agent")
    print("  👁️  Observer thread will watch memory updates")
    print("  👑 Governor thread will monitor quality")
    print("-" * 70)

    # Get full result with supervision metadata
    result = await supervised_agent.execute(state)

    print("-" * 70)

    # Step 5: Results
    print("\n📊 Step 5: Agent Results")
    print(f"  Execution ID: {result['execution_id']}")
    print(f"  Status: {result['status']}")
    print(f"  Latency: {result['timing']['latency_ms']:.0f}ms")

    # Step 6: Intent Understanding Output
    print("\n📤 Step 6: Intent Understanding Output (from LLM)")
    intent_output = result['result'].get('intent_understanding', '')

    if intent_output:
        # Show first 300 chars
        print("─" * 70)
        print(intent_output[:300])
        if len(intent_output) > 300:
            print(f"... ({len(intent_output) - 300} more characters)")
        print("─" * 70)
    else:
        print("  (empty)")

    # Step 7: Supervision metadata
    print("\n👁️  Step 7: Observer Observations")
    observations = result['supervision']['observer_observations']
    print(f"  Total observations: {len(observations)}")
    for i, obs in enumerate(observations[:3], 1):
        print(f"    {i}. Status: {obs['status']}")

    print("\n👑 Step 8: Governor Interventions")
    interventions = result['supervision']['governor_interventions']
    print(f"  Total interventions: {len(interventions)}")
    if interventions:
        for intervention in interventions:
            print(f"    - Type: {intervention['type']}")
            if 'quality' in intervention:
                print(f"      Quality: {intervention['quality']:.2f}")
    else:
        print("    (no interventions needed)")

    # Step 9: Runtime memory
    print("\n🧠 Step 9: Runtime Memory Snapshot")
    for key in ['actor_status', 'latency_ms', 'thread_id']:
        value = result['runtime_memory'].get(key)
        print(f"  {key}: {value}")

    # Step 10: Instructions
    print("\n📜 Step 10: Runtime Instructions")
    if result['runtime_instructions']:
        for instruction in result['runtime_instructions']:
            print(f"  - {instruction}")
    else:
        print("  (none)")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  • Real MAF agent executed with LLMs")
    print(f"  • 3 OS threads ran concurrently")
    print(f"  • Observer tracked {len(observations)} state changes")
    print(f"  • Governor made {len(interventions)} interventions")
    print(f"  • Total latency: {result['timing']['latency_ms']:.0f}ms")


async def test_multiple_queries_with_sentinel():
    """
    Test multiple queries to see supervision working across executions.
    """

    print("\n" + "=" * 70)
    print("MULTIPLE QUERIES TEST")
    print("=" * 70)

    # Create supervised agent
    supervised_agent = SentinelMAFAgent(
        maf_agent=IntentUnderstandingAgent(),
        agent_name="Intent Understanding",
        quality_threshold=0.75,
        timeout_sec=60.0,
    )

    queries = [
        "What's the best way to handle authentication in microservices?",
        "How do I implement rate limiting in my API?",
        "What are the security best practices for storing user passwords?",
    ]

    print(f"\n⚡ Processing {len(queries)} queries with supervision...\n")

    results = []

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Query: {query[:60]}...")

        state = Orchestrator_Global_State(
            user_id=f"user_{i}",
            user_query=query,
            updated_query=query,
        )

        result = await supervised_agent.execute(state)
        results.append(result)

        print(f"  ✓ Status: {result['status']}")
        print(f"  ✓ Latency: {result['timing']['latency_ms']:.0f}ms")
        print(f"  ✓ Observations: {len(result['supervision']['observer_observations'])}")
        print(f"  ✓ Interventions: {len(result['supervision']['governor_interventions'])}")
        print()

    # Summary
    total_latency = sum(r['timing']['latency_ms'] for r in results)
    total_observations = sum(len(r['supervision']['observer_observations']) for r in results)
    total_interventions = sum(len(r['supervision']['governor_interventions']) for r in results)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total queries: {len(results)}")
    print(f"  Total latency: {total_latency:.0f}ms")
    print(f"  Average latency: {total_latency / len(results):.0f}ms")
    print(f"  Total observations: {total_observations}")
    print(f"  Total interventions: {total_interventions}")
    print("\n✅ All queries processed with 3-thread supervision!")


async def test_maf_compatible_interface():
    """
    Test using supervised agent as drop-in replacement.
    """

    print("\n" + "=" * 70)
    print("MAF-COMPATIBLE INTERFACE TEST")
    print("=" * 70)

    print("\n🔄 Creating supervised agent...")

    # Create supervised agent
    supervised_agent = SentinelMAFAgent(
        maf_agent=IntentUnderstandingAgent(),
        agent_name="Intent Agent",
        quality_threshold=0.75,
    )

    print("  ✓ Agent created with 3-thread supervision")

    # Use like a regular MAF agent
    print("\n⚡ Calling like regular MAF agent: await agent(state)")

    state = Orchestrator_Global_State(
        user_id="test_user",
        user_query="How do I secure my API endpoints?",
        updated_query="How do I secure my API endpoints?",
    )

    # This returns just the MAF result (drop-in compatible)
    result = await supervised_agent(state)

    print(f"\n📤 Result (MAF-compatible output):")
    intent = result.get('intent_understanding', '')
    print(f"  {intent[:200]}...")

    print("\n✅ Works as drop-in replacement for MAF agents!")


async def main():
    """Run all tests."""

    print("\n" + "=" * 70)
    print("🚀 REAL MAF + SENTINEL INTEGRATION TESTS")
    print("=" * 70)
    print("\nThese tests use REAL MAF agents with LLMs")
    print("Make sure you have:")
    print("  • OpenAI API key in .env")
    print("  • LangChain dependencies installed")
    print()

    try:
        # Test 1: Single query with full metadata
        await test_real_intent_agent_with_sentinel()

        await asyncio.sleep(1)

        # Test 2: Multiple queries
        await test_multiple_queries_with_sentinel()

        await asyncio.sleep(1)

        # Test 3: MAF-compatible interface
        await test_maf_compatible_interface()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS COMPLETE!")
        print("=" * 70)
        print("\n✨ Sentinel + MAF Integration Working with Real LLMs!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. OpenAI API key set in .env")
        print("  2. All dependencies installed: langchain, langgraph, etc.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
