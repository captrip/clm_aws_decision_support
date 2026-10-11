"""
Quick test with a real user query
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
from MAF.demo_supervised_agent import SupervisedIntentUnderstandingAgent


async def test_single_query(query: str):
    """Test a single real query."""
    print(f"\n{'=' * 70}")
    print(f"Testing Real User Query")
    print(f"{'=' * 70}")
    print(f"\n💬 Query: {query}")

    # Create supervised agent
    agent = SupervisedIntentUnderstandingAgent()

    # Process query
    print(f"\n🚀 Processing with full supervision...\n")

    state = {
        "user_query": query,
        "updated_query": query,
        "user_id": "test_user",
    }

    result = await agent(state)

    # Display detailed results
    print(f"\n{'=' * 70}")
    print(f"Results")
    print(f"{'=' * 70}")
    print(f"\n📊 Intent Analysis:")
    print(f"  {result['intent_understanding']}")
    print(f"\n📈 Metrics:")
    print(f"  Confidence: {result.get('confidence', 'N/A')}")
    print(f"  Processed Query: {result.get('processed_query', 'N/A')}")
    print(f"\n🔍 Supervision Stats:")
    print(f"  Executions: {agent.execution_count}")
    print(f"  Interventions: {agent.intervention_count}")
    print(f"  State Status: {agent.state.get('status')}")

    # Show metrics collected
    print(f"\n📉 Performance Metrics:")
    metrics = agent.state.get("metrics", {})
    for key, values in metrics.items():
        if values:
            if isinstance(values, list):
                avg = sum(values) / len(values)
                print(f"  {key}: {avg:.2f} (avg over {len(values)} samples)")
            else:
                print(f"  {key}: {values}")


if __name__ == "__main__":
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Can you help me build a multi-agent system with supervision?"

    asyncio.run(test_single_query(query))
