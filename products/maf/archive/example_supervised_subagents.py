"""
Minimal Example: Supervised MAF Subagent

Shows how to wrap a MAF subagent with supervision in 5 lines.
"""

import sys
from pathlib import Path

# Add workspace root to path for imports
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
from typing import Dict, Any


async def example_supervised_intent_agent():
    """Example: Wrap IntentUnderstandingAgent with supervision."""
    
    # Import your MAF agent
    # from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent
    # from MAF.supervised_subagents import wrap_maf_subagent
    
    # For demonstration:
    class MockIntentAgent:
        async def ainvoke(self, state):
            await asyncio.sleep(0.1)
            return {
                "intent": "user_query" if state.get("query") else "greeting",
                "confidence": 0.95,
            }
    
    agent = MockIntentAgent()
    
    # Wrap with supervision (1 line)
    from MAF.supervised_subagents import wrap_maf_subagent
    
    result = await wrap_maf_subagent(
        agent_instance=agent,
        agent_type="intent",
        input_data={"query": "What's the weather?"},
    )
    
    print(f"\n📤 Final result: {result}")
    return result


async def example_orchestrator_with_supervision():
    """Example: Use supervised orchestrator."""
    
    from MAF.orchestrator_integration import orchestrator_with_supervision
    
    # Mock orchestrator
    class MockOrchestrator:
        class MockAgent:
            async def ainvoke(self, state):
                await asyncio.sleep(0.05)
                return {"dummy": "result"}
        
        def __init__(self):
            self.intent_agent = self.MockAgent()
            self.planner_agent = self.MockAgent()
            self.skill_executor = self.MockAgent()
            self.response_generator = self.MockAgent()
    
    orchestrator = MockOrchestrator()
    
    result = await orchestrator_with_supervision(
        user_query="Tell me a joke",
        memory_info={"context": "user_session_1"},
        use_supervision=True,
        orchestrator_instance=orchestrator,
    )
    
    print(f"\n📊 Orchestrator result keys: {list(result.keys())}")
    return result


async def example_enable_supervision():
    """Example: Enable supervision in existing orchestrator."""
    
    from MAF.orchestrator_integration import enable_supervision_in_orchestrator
    
    # Mock orchestrator
    class MockOrchestrator:
        class MockAgent:
            async def ainvoke(self, state):
                return {"status": "ok"}
        
        def __init__(self):
            self.intent_agent = self.MockAgent()
            self.planner_agent = self.MockAgent()
            self.skill_executor = self.MockAgent()
            self.response_generator = self.MockAgent()
        
        async def ainvoke(self, state):
            return {"original": True}
    
    orchestrator = MockOrchestrator()
    
    # Enable supervision
    enable_supervision_in_orchestrator(orchestrator)
    
    # Now calls use supervision automatically
    result = await orchestrator.ainvoke({
        "user_query": "Hello",
        "memory_info": {},
    })
    
    print(f"\n✅ Orchestrator supervised: {result.get('supervised', False)}")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("MAF Supervised Subagent Examples")
    print("=" * 60)
    
    print("\n1️⃣  Example: Wrap Single Subagent")
    asyncio.run(example_supervised_intent_agent())
    
    print("\n" + "=" * 60)
    print("2️⃣  Example: Supervised Orchestrator")
    asyncio.run(example_orchestrator_with_supervision())
    
    print("\n" + "=" * 60)
    print("3️⃣  Example: Enable Supervision in Existing Orchestrator")
    asyncio.run(example_enable_supervision())
    
    print("\n" + "=" * 60)
    print("✅ All examples complete")
