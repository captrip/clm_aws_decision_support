"""
Example: Using Sentinel-Wrapped MAF Agents in Orchestrator

This shows how to integrate the 3-thread architecture into your
existing MAF orchestrator with minimal changes.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
from typing import Dict, Any

# Your existing MAF imports (commented out - uncomment when ready)
from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent, ActionExtractorAgent
from MAF.MAFI.agents.action_planner import ActionPlannerAgent
from MAF.global_.orchestrator_graph import Global_Orchestrator
from MAF.global_.state import Orchestrator_Global_State

# Sentinel integration
from MAF.sentinel_maf_integration import wrap_maf_agent_with_sentinel

# Mock for demo
# from MAF.test_supervised_agents import MockIntentUnderstandingAgent


# ==========================================
# Pattern 1: Wrap Individual Agents
# ==========================================

async def pattern_1_wrap_individual_agents():
    """
    Pattern 1: Wrap specific agents you want to supervise.

    Use this when you want fine-grained control over which agents
    run with supervision.
    """

    print("\n" + "=" * 70)
    print("Pattern 1: Wrap Individual Agents")
    print("=" * 70)

    # Original MAF agents
    print("\n📦 Creating MAF agents...")

    intent_agent = IntentUnderstandingAgent()  # Real MAF agent
    # intent_agent = MockIntentUnderstandingAgent()  # Mock for demo

    planner_agent = ActionPlannerAgent()  # Real MAF agent
    # execution_agent = SkillExecutionAgent()  # Real MAF agent

    # Wrap with Sentinel
    print("\n🔄 Wrapping with Sentinel (3-thread supervision)...")

    supervised_intent = wrap_maf_agent_with_sentinel(
        maf_agent=intent_agent,
        agent_name="Intent Understanding",
        quality_threshold=0.75,
        timeout_sec=30.0,
    )

    # supervised_planner = wrap_maf_agent_with_sentinel(
    #     maf_agent=planner_agent,
    #     agent_name="Action Planning",
    #     quality_threshold=0.70,
    # )

    print("  ✓ Intent agent wrapped")
    # print("  ✓ Planner agent wrapped")

    # Use in your orchestrator flow
    print("\n⚡ Executing with supervision...")

    state = {
        "user_query": "How do I implement authentication?",
        "updated_query": "How do I implement authentication?",
        "user_id": "user_123",
    }

    # This now runs with Actor|Observer|Governor threads
    result = await supervised_intent(state)

    print(f"\n📊 Result:")
    print(f"  {result}")

    print("\n✅ Pattern 1 Complete!")


# ==========================================
# Pattern 2: Orchestrator Factory
# ==========================================

def create_supervised_orchestrator():
    """
    Pattern 2: Factory function that creates an orchestrator
    with all agents pre-wrapped with Sentinel.

    Use this for a clean, reusable setup.
    """

    print("\n" + "=" * 70)
    print("Pattern 2: Orchestrator Factory")
    print("=" * 70)

    # Uncomment when ready to use real MAF agents:
    
    from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent
    from MAF.MAFI.agents.action_planner import ActionPlannerAgent

    return {
        "intent": wrap_maf_agent_with_sentinel(
            IntentUnderstandingAgent(),
            "Intent Understanding",
            quality_threshold=0.75,
        ),
        "planner": wrap_maf_agent_with_sentinel(
            ActionPlannerAgent(),
            "Action Planning",
            quality_threshold=0.70,
        ),
        # Add more agents...
    }
    

    # Demo with mock agents
    # return {
    #     "intent": wrap_maf_agent_with_sentinel(
    #         MockIntentUnderstandingAgent(),
    #         "Intent Understanding",
    #         quality_threshold=0.75,
    #     ),
    # }


async def pattern_2_use_factory():
    """Use the factory."""

    agents = create_supervised_orchestrator()

    print(f"\n📦 Created orchestrator with {len(agents)} supervised agents")

    state = {
        "user_query": "Test query",
        "updated_query": "Test query",
    }

    result = await agents["intent"](state)

    print(f"✅ Pattern 2 Complete!")


# ==========================================
# Pattern 3: Drop-in Replacement
# ==========================================

async def pattern_3_drop_in_replacement():
    """
    Pattern 3: Replace agents in existing orchestrator.

    Minimal changes to your existing code.
    """

    print("\n" + "=" * 70)
    print("Pattern 3: Drop-in Replacement in Existing Orchestrator")
    print("=" * 70)

    # Your existing orchestrator setup:
    """
    orc = Global_Orchestrator(
        auto_load_skills=True,
        db_path="subgraph_registry.db",
    ).build()
    """

    # To add supervision, just wrap the agents:
    """
    # Option A: Wrap after build
    orc.intent_agent = wrap_maf_agent_with_sentinel(
        orc.intent_agent,
        "Intent Understanding",
        quality_threshold=0.75,
    )

    # Option B: Pass wrapped agents at construction
    intent_agent = wrap_maf_agent_with_sentinel(
        IntentUnderstandingAgent(),
        "Intent Understanding",
    )

    orc = Global_Orchestrator(
        intent_agent=intent_agent,  # Use wrapped version
        # ... other params
    ).build()
    """

    print("\n✅ Pattern 3 Example (see comments for real usage)")


# ==========================================
# Pattern 4: Conditional Supervision
# ==========================================

async def pattern_4_conditional_supervision():
    """
    Pattern 4: Enable/disable supervision based on config.

    Use this for development vs production, or per-user settings.
    """

    print("\n" + "=" * 70)
    print("Pattern 4: Conditional Supervision")
    print("=" * 70)

    # Config
    ENABLE_SUPERVISION = True  # Could come from env var or config file

    # Create agent
    intent_agent = MockIntentUnderstandingAgent()

    # Conditionally wrap
    if ENABLE_SUPERVISION:
        print("\n🔄 Supervision ENABLED")
        agent = wrap_maf_agent_with_sentinel(
            intent_agent,
            "Intent Understanding",
            quality_threshold=0.75,
        )
    else:
        print("\n🎭 Supervision DISABLED (using raw agent)")
        agent = intent_agent

    # Use normally
    state = {
        "user_query": "Test",
        "updated_query": "Test",
    }

    result = await agent(state)

    print("\n✅ Pattern 4 Complete!")


# ==========================================
# Pattern 5: Collect Supervision Metrics
# ==========================================

async def pattern_5_collect_metrics():
    """
    Pattern 5: Collect and aggregate supervision metrics.

    Use this for monitoring, logging, or analytics.
    """

    print("\n" + "=" * 70)
    print("Pattern 5: Collect Supervision Metrics")
    print("=" * 70)

    from MAF.sentinel_maf_integration import SentinelMAFAgent

    # Create agent
    agent = SentinelMAFAgent(
        maf_agent=MockIntentUnderstandingAgent(),
        agent_name="Intent Agent",
        quality_threshold=0.7,
    )

    # Execute
    state = {"user_query": "Test", "updated_query": "Test"}

    # Use .execute() to get full metadata (not just result)
    result = await agent.execute(state)

    # Extract metrics
    metrics = {
        "execution_id": result["execution_id"],
        "latency_ms": result["timing"]["latency_ms"],
        "status": result["status"],
        "observations_count": len(result["supervision"]["observer_observations"]),
        "interventions_count": len(result["supervision"]["governor_interventions"]),
    }

    print(f"\n📊 Collected Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # You could send these to:
    # - Logging system
    # - Monitoring dashboard (Prometheus, Grafana)
    # - Database for analytics
    # - Real-time alerting system

    print("\n✅ Pattern 5 Complete!")


# ==========================================
# Example: Complete Orchestrator Flow
# ==========================================

async def example_complete_flow():
    """
    Example: Complete orchestrator flow with supervision.

    This simulates a full MAF orchestration with all agents supervised.
    """

    print("\n" + "=" * 70)
    print("COMPLETE ORCHESTRATOR FLOW WITH SUPERVISION")
    print("=" * 70)

    # Step 1: Create supervised agents
    print("\n📦 Step 1: Create Supervised Agents")

    agents = {
        "intent": wrap_maf_agent_with_sentinel(
            MockIntentUnderstandingAgent(),
            "Intent Understanding",
            quality_threshold=0.75,
        ),
        # Add more agents here when ready
    }

    print(f"  ✓ Created {len(agents)} supervised agents")

    # Step 2: User input
    print("\n💬 Step 2: User Query")
    user_query = "How do I build a microservices architecture with monitoring?"
    print(f"  Query: {user_query}")

    # Step 3: Execute orchestration
    print("\n⚡ Step 3: Execute Orchestration")
    print("-" * 70)

    # Intent Understanding (with supervision)
    print("\n🎯 Stage 1: Intent Understanding")
    intent_state = {
        "user_query": user_query,
        "updated_query": user_query,
        "user_id": "user_123",
    }

    from MAF.sentinel_maf_integration import SentinelMAFAgent
    if isinstance(agents["intent"], SentinelMAFAgent):
        intent_result = await agents["intent"].execute(intent_state)

        print(f"  Status: {intent_result['status']}")
        print(f"  Latency: {intent_result['timing']['latency_ms']:.0f}ms")
        print(f"  Observations: {len(intent_result['supervision']['observer_observations'])}")
        print(f"  Interventions: {len(intent_result['supervision']['governor_interventions'])}")

        # Extract the actual result
        intent_understanding = intent_result["result"]
    else:
        intent_understanding = await agents["intent"](intent_state)

    print(f"  Intent: {str(intent_understanding)[:60]}...")

    # Step 4: Collect all metrics
    print("\n📊 Step 4: Aggregate Supervision Metrics")

    all_metrics = {
        "intent": {
            "latency_ms": intent_result["timing"]["latency_ms"],
            "observations": len(intent_result["supervision"]["observer_observations"]),
            "interventions": len(intent_result["supervision"]["governor_interventions"]),
            "status": intent_result["status"],
        },
        # Add metrics from other agents here
    }

    print(f"  Total agents: {len(all_metrics)}")
    print(f"  All completed: {all(m['status'] == 'COMPLETED' for m in all_metrics.values())}")

    print("\n" + "=" * 70)
    print("✅ COMPLETE FLOW EXECUTED!")
    print("=" * 70)


# ==========================================
# Main
# ==========================================

async def main():
    """Run all examples."""

    examples = [
        ("Pattern 1", pattern_1_wrap_individual_agents),
        ("Pattern 2", pattern_2_use_factory),
        ("Pattern 3", pattern_3_drop_in_replacement),
        ("Pattern 4", pattern_4_conditional_supervision),
        ("Pattern 5", pattern_5_collect_metrics),
        ("Complete Flow", example_complete_flow),
    ]

    for name, example in examples:
        await example()
        await asyncio.sleep(0.3)

    print("\n" + "=" * 70)
    print("🎉 ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\n📚 Summary:")
    print("  - Pattern 1: Wrap individual agents selectively")
    print("  - Pattern 2: Factory function for clean setup")
    print("  - Pattern 3: Drop-in replacement in existing code")
    print("  - Pattern 4: Conditional supervision (dev/prod)")
    print("  - Pattern 5: Collect and monitor metrics")
    print("\n✨ Choose the pattern that fits your use case!")


if __name__ == "__main__":
    asyncio.run(main())
