"""
MAF Orchestrator Integration with Supervised Subagents

Minimal changes to orchestrator to use supervised subagents.
"""

import sys
from pathlib import Path

# Add workspace root to path for imports
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

from typing import Optional, Dict, Any
import asyncio

# Example: How to modify Global_Orchestrator to use supervised subagents


async def orchestrator_with_supervision(
    user_query: str,
    memory_info: Dict[str, Any],
    state_manager: Optional[Any] = None,
    use_supervision: bool = True,
    orchestrator_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Modified orchestrator flow that uses supervised subagents.
    
    Usage in your existing orchestrator:
    
        # In Global_Orchestrator.__init__ or similar:
        self.use_supervision = True
        
        # In your ainvoke or execute method:
        if self.use_supervision:
            result = await orchestrator_with_supervision(
                user_query=user_query,
                memory_info=memory_info,
                state_manager=self.state_manager,
                orchestrator_instance=self,
            )
        else:
            result = await original_execute(...)
    """
    
    from MAF.supervised_subagents import wrap_maf_subagent
    
    if not orchestrator_instance:
        raise ValueError("orchestrator_instance required")
    
    print(f"\n🔄 Orchestrator: Starting with supervision")
    
    # Stage 1: Intent Understanding (SUPERVISED)
    print("\n📊 Stage 1: Intent Understanding")
    
    intent_input = {
        "user_query": user_query,
        "memory_info": memory_info,
    }
    
    if use_supervision:
        intent_result = await wrap_maf_subagent(
            agent_instance=orchestrator_instance.intent_agent,
            agent_type="intent",
            input_data=intent_input,
        )
    else:
        intent_result = await orchestrator_instance.intent_agent.ainvoke(intent_input)
    
    print(f"✓ Intent: {intent_result.get('intent', 'unknown')}")
    
    # Stage 2: Action Planning (SUPERVISED)
    print("\n📊 Stage 2: Action Planning")
    
    planning_input = {
        "intent": intent_result,
        "memory_info": memory_info,
    }
    
    if use_supervision:
        planning_result = await wrap_maf_subagent(
            agent_instance=orchestrator_instance.planner_agent,
            agent_type="planning",
            input_data=planning_input,
        )
    else:
        planning_result = await orchestrator_instance.planner_agent.ainvoke(planning_input)
    
    print(f"✓ Plan: {len(planning_result.get('steps', []))} steps")
    
    # Stage 3: Execute Skills (can add supervision later)
    print("\n📊 Stage 3: Skill Execution")
    
    execution_input = {
        "plan": planning_result,
        "memory_info": memory_info,
    }
    
    execution_result = await orchestrator_instance.skill_executor.ainvoke(execution_input)
    
    print(f"✓ Execution: {execution_result.get('status', 'unknown')}")
    
    # Stage 4: Generate Response
    print("\n📊 Stage 4: Response Generation")
    
    response_input = {
        "execution_result": execution_result,
        "memory_info": memory_info,
    }
    
    response_result = await orchestrator_instance.response_generator.ainvoke(response_input)
    
    print(f"✓ Response ready")
    
    # Aggregate results
    final_result = {
        "user_query": user_query,
        "intent": intent_result,
        "plan": planning_result,
        "execution": execution_result,
        "response": response_result,
        "supervised": use_supervision,
    }
    
    return final_result


# ==========================================
# Drop-in Integration Point
# ==========================================

def enable_supervision_in_orchestrator(orchestrator_instance):
    """
    Enable supervision in existing orchestrator with minimal changes.
    
    Usage:
        from MAF.orchestrator_integration import enable_supervision_in_orchestrator
        
        orchestrator = Global_Orchestrator()
        enable_supervision_in_orchestrator(orchestrator)
        
        # Now orchestrator uses supervision automatically
        result = await orchestrator.ainvoke(state)
    """
    
    # Store original ainvoke
    original_ainvoke = orchestrator_instance.ainvoke
    
    # Replace with supervised version
    async def ainvoke_supervised(state):
        return await orchestrator_with_supervision(
            user_query=state.get("user_query", ""),
            memory_info=state.get("memory_info", {}),
            state_manager=getattr(orchestrator_instance, "state_manager", None),
            use_supervision=True,
            orchestrator_instance=orchestrator_instance,
        )
    
    orchestrator_instance.ainvoke = ainvoke_supervised
    orchestrator_instance._original_ainvoke = original_ainvoke
    orchestrator_instance.use_supervision = True
    
    print("✅ Supervision enabled in orchestrator")


def disable_supervision_in_orchestrator(orchestrator_instance):
    """Disable supervision and revert to original behavior."""
    if hasattr(orchestrator_instance, "_original_ainvoke"):
        orchestrator_instance.ainvoke = orchestrator_instance._original_ainvoke
        orchestrator_instance.use_supervision = False
        print("✅ Supervision disabled in orchestrator")
