"""
Integration example showing how to use supervised agents in the MAF notebook workflow.

This shows how to replace the regular agent with a supervised version
in the existing test.ipynb workflow.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

# This demonstrates the pattern shown in test.ipynb but with supervision


async def notebook_example_without_supervision():
    """
    Original notebook pattern (from test.ipynb):

    ```python
    from global_.orchestrator_graph import Global_Orchestrator
    from global_.state import Orchestrator_Global_State
    from MAFI.agent_factory.graph_factory_concurrent import SubgraphFactory

    orc = Global_Orchestrator(
        auto_load_skills=True,
        db_path="subgraph_registry.db",
        SubgraphFactory=SubgraphFactory,
    ).build()

    initial_state = Orchestrator_Global_State(
        user_id="test_123",
        user_query="hi"
    )

    result = await orc.ainvoke(initial_state)
    print(result['final_answer'])
    ```
    """
    print("Original pattern (without supervision) - see test.ipynb")


async def notebook_example_with_supervision():
    """
    Modified pattern WITH supervision at the agent level.

    To add supervision to individual agents in the orchestrator:

    ```python
    from global_.orchestrator_graph import Global_Orchestrator
    from global_.state import Orchestrator_Global_State
    from MAFI.agent_factory.graph_factory_concurrent import SubgraphFactory
    from MAF.test_real_supervised_agents import SupervisedIntentUnderstandingAgent
    from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent

    # Step 1: Create the orchestrator
    orc = Global_Orchestrator(
        auto_load_skills=True,
        db_path="subgraph_registry.db",
        SubgraphFactory=SubgraphFactory,
    ).build()

    # Step 2: Replace specific agents with supervised versions
    # (This depends on how the orchestrator exposes its agents)
    # Option A: If you can inject agents at build time
    supervised_iu = SupervisedIntentUnderstandingAgent(
        IntentUnderstandingAgent()
    )

    # Option B: Wrap the entire orchestrator call with monitoring
    # (shown in the full example below)

    # Step 3: Use normally
    initial_state = Orchestrator_Global_State(
        user_id="test_123",
        user_query="hi"
    )

    result = await orc.ainvoke(initial_state)
    print(result['final_answer'])

    # Step 4: Access supervision metrics
    print(f"Agent executions: {supervised_iu.execution_count}")
    print(f"Interventions: {supervised_iu.intervention_count}")
    ```
    """
    print("Pattern with supervision - see example above")


def show_integration_patterns():
    """
    Shows different ways to integrate supervision into MAF.
    """
    print("\n" + "=" * 70)
    print("MAF Supervision Integration Patterns")
    print("=" * 70)

    print("\n📋 Pattern 1: Wrap Individual Agent")
    print("-" * 70)
    print("""
    # Import the supervised wrapper
    from MAF.test_real_supervised_agents import SupervisedIntentUnderstandingAgent
    from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent

    # Create supervised version
    original_agent = IntentUnderstandingAgent()
    supervised_agent = SupervisedIntentUnderstandingAgent(original_agent)

    # Use in orchestrator
    # (requires modifying orchestrator to accept custom agents)
    """)

    print("\n📋 Pattern 2: Test Individual Agent Before Integration")
    print("-" * 70)
    print("""
    # Test the supervised agent standalone
    from MAF.test_real_supervised_agents import SupervisedIntentUnderstandingAgent
    from MAF.global_.state import Orchestrator_Global_State

    supervised = SupervisedIntentUnderstandingAgent()

    state = Orchestrator_Global_State(
        user_id="test_user",
        user_query="Your query here"
    )

    result = await supervised(state)

    # Check metrics
    print(f"Quality: {supervised.state.get('metrics')}")
    print(f"Executions: {supervised.execution_count}")
    """)

    print("\n📋 Pattern 3: Notebook Cell Integration")
    print("-" * 70)
    print("""
    # In Jupyter notebook (test.ipynb):

    # Cell 1: Setup
    from MAF.test_real_supervised_agents import SupervisedIntentUnderstandingAgent
    from MAF.global_.state import Orchestrator_Global_State

    # Cell 2: Create agent
    supervised_agent = SupervisedIntentUnderstandingAgent()

    # Cell 3: Test with query
    state = Orchestrator_Global_State(
        user_id="notebook_user",
        user_query="What's the weather?"
    )
    result = await supervised_agent(state)

    # Cell 4: View results
    print(result['intent_understanding'])
    print(f"Quality Score: {supervised_agent.state.get('metrics')}")
    """)

    print("\n📋 Pattern 4: Full Orchestrator with Metrics Collection")
    print("-" * 70)
    print("""
    # Wrap orchestrator call to collect agent metrics
    from global_.orchestrator_graph import Global_Orchestrator

    class MetricsCollector:
        def __init__(self, orchestrator):
            self.orchestrator = orchestrator
            self.agent_metrics = {}

        async def ainvoke(self, state):
            # Intercept and collect metrics
            result = await self.orchestrator.ainvoke(state)
            return result

    orc = Global_Orchestrator(...).build()
    monitored_orc = MetricsCollector(orc)

    result = await monitored_orc.ainvoke(state)
    """)

    print("\n" + "=" * 70)
    print("Choose the pattern that fits your use case!")
    print("=" * 70)


if __name__ == "__main__":
    show_integration_patterns()

    print("\n\n🚀 To run the REAL supervised agent test:")
    print("   python MAF/test_real_supervised_agents.py \"your query here\"")

    print("\n📓 To use in Jupyter notebook (test.ipynb):")
    print("   Copy Pattern 3 above into a notebook cell")
