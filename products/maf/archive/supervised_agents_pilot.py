"""
MAF Supervised Agents Pilot

Real integration of supervision with MAF subagents.
This implements a supervised wrapper for IntentUnderstandingAgent as the first pilot.

Phase 1 Implementation: IntentUnderstandingAgent
"""

import sys
from pathlib import Path

# Add workspace root to path for imports
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
import threading
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass

from experiment.agent_supervision import (
    SupervisedAgentState,
    AgentStatus,
    CorrectionType,
    InterventionRequest,
)
from MAF.supervised_subagents import SubagentObserver, SubagentGovernor


class SupervisedIntentUnderstandingAgent:
    """
    Wraps MAF IntentUnderstandingAgent with full supervision.
    
    Compatible drop-in replacement - same interface as original agent.
    """
    
    def __init__(self, original_agent: Any = None):
        """
        Args:
            original_agent: IntentUnderstandingAgent instance to wrap.
                           If None, will import and create one.
        """
        if original_agent is None:
            # Import and create the real agent
            from MAFI.agents.IU_agents import IntentUnderstandingAgent
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
        """
        Execute with supervision - same interface as original agent.
        """
        self.execution_count += 1
        
        self.state.update("status", AgentStatus.WORKING)
        self.state.update("current_operation", "Intent Understanding")
        
        # Start observer and governor
        self.observer = SubagentObserver(self.state, "intent")
        self.governor = GovernorIntentSpecific(self.state, "intent")
        
        self.observer.start()
        self.governor.start()
        
        try:
            # Check for pre-execution interventions
            intervention = self.state.pop_intervention()
            if intervention:
                print(f"⚠️ [Intent Agent] Pre-exec intervention: {intervention.correction_type.value}")
            
            # Execute original agent
            start_time = time.time()
            result = await self.original_agent(state)
            latency_ms = (time.time() - start_time) * 1000
            
            # Compute quality metrics
            quality_score = self._compute_quality(result)
            
            # Update metrics
            self.state.add_metric("latency_ms", latency_ms)
            self.state.add_metric("quality_score", quality_score)
            
            # Check for post-execution interventions
            intervention = self.state.pop_intervention()
            if intervention:
                self.intervention_count += 1
                print(f"⚠️ [Intent Agent] Intervention #{self.intervention_count}: {intervention.correction_type.value}")
                result = self._apply_intervention(intervention, result, state)
            
            self.state.update("status", AgentStatus.COMPLETE)
            
            print(f"✅ [Intent Agent] Quality: {quality_score:.2f} | Latency: {latency_ms:.0f}ms")
            
            return result
        
        except Exception as e:
            self.state.update("status", AgentStatus.ERROR)
            self.state.add_error(str(e))
            print(f"❌ [Intent Agent] Error: {e}")
            raise
        
        finally:
            # Stop monitoring threads
            if self.observer:
                self.observer.stop()
            if self.governor:
                self.governor.stop()
            if self.observer:
                self.observer.join(timeout=1)
            if self.governor:
                self.governor.join(timeout=1)
    
    def _compute_quality(self, result: Dict[str, Any]) -> float:
        """
        Compute quality score for intent understanding output.
        
        Returns: 0.0-1.0 quality score
        """
        intent_text = result.get("intent_understanding", "")
        
        # Quality checks
        if not intent_text:
            return 0.2
        
        if isinstance(intent_text, str) and len(intent_text.strip()) > 0:
            return 0.9  # Good output
        
        return 0.5
    
    def _apply_intervention(self, intervention: InterventionRequest, result: Any, state: Dict[str, Any]) -> Any:
        """Apply governance intervention to result."""
        
        if intervention.correction_type == CorrectionType.RETRY:
            print(f"   → Would retry with same input")
            # In real scenario, could re-run the agent
        
        elif intervention.correction_type == CorrectionType.MODIFY_PARAMS:
            print(f"   → Would modify params: {intervention.parameters}")
        
        elif intervention.correction_type == CorrectionType.ESCALATE:
            print(f"   → Escalating for manual review")
        
        return result


class GovernorIntentSpecific(SubagentGovernor):
    """
    Governor specialized for Intent Understanding Agent.
    
    Rules specific to intent understanding quality.
    """
    
    def _apply_subagent_rules(self, snapshot: Dict[str, Any]):
        """Apply rules specific to intent understanding."""
        metrics = snapshot.get("metrics", {})
        quality = metrics.get("quality_score", 1.0)
        latency = metrics.get("latency_ms", 0)
        
        # RULE 1: Empty or very short intent (quality indicator)
        if quality < 0.5:
            print(f"👑 [Intent] RULE 1: Low quality output ({quality:.2f})")
            self._request_intervention(
                CorrectionType.RETRY,
                f"Intent quality {quality:.2f} below threshold",
                priority=3
            )
        
        # RULE 2: Slow intent understanding (possible hallucination/reasoning loop)
        if latency > 3000:
            print(f"👑 [Intent] RULE 2: Slow intent processing ({latency:.0f}ms)")
            self._request_intervention(
                CorrectionType.MODIFY_PARAMS,
                "Intent extraction taking too long",
                parameters={"timeout": 5000, "model": "faster"},
                priority=2
            )
        
        # RULE 3: Multiple retries indicate fundamental issue
        decisions = snapshot.get("decisions", [])
        retry_count = sum(1 for d in decisions if "Retry" in d)
        if retry_count > 1:
            print(f"👑 [Intent] RULE 3: Multiple retries ({retry_count})")
            self._request_intervention(
                CorrectionType.ESCALATE,
                f"Intent failed {retry_count} times - escalating",
                priority=5
            )


# ==========================================
# Integration Helper
# ==========================================

def create_supervised_intent_agent(original_agent: Optional[Any] = None) -> SupervisedIntentUnderstandingAgent:
    """
    Factory function to create a supervised intent agent.
    
    Usage:
        supervised_agent = create_supervised_intent_agent()
        result = await supervised_agent(state)
    """
    return SupervisedIntentUnderstandingAgent(original_agent)


# ==========================================
# Demo: Test with MAF orchestrator
# ==========================================

async def demo_supervised_orchestrator():
    """
    Demo: Use supervised IntentUnderstandingAgent in MAF orchestrator.
    """
    print("\n" + "=" * 60)
    print("DEMO: Supervised IntentUnderstandingAgent in MAF")
    print("=" * 60)
    
    # Import orchestrator
    from global_.orchestrator_graph import Global_Orchestrator
    
    # Create orchestrator
    orchestrator = Global_Orchestrator()
    
    # Replace with supervised agent
    print("\n📦 Creating supervised intent agent...")
    supervised_intent = create_supervised_intent_agent(orchestrator.intent_understanding)
    orchestrator.intent_understanding = supervised_intent
    
    # Test state
    test_state = {
        "user_query": "What's the weather in San Francisco?",
        "updated_query": "What's the weather in San Francisco?",
        "user_id": "user_123",
        "session_id": "session_456",
    }
    
    print("\n🚀 Executing orchestrator with supervised intent agent...")
    
    # Get the intent understanding subgraph (if available)
    try:
        result = await orchestrator.intent_understanding(test_state)
        print(f"\n📤 Intent Result: {result}")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def demo_simple_supervised_intent():
    """
    Simple demo: Just test the supervised intent agent alone.
    """
    print("\n" + "=" * 60)
    print("DEMO: Supervised Intent Agent (Standalone)")
    print("=" * 60)
    
    # Create supervised agent
    print("\n📦 Creating supervised intent agent...")
    agent = create_supervised_intent_agent()
    
    # Test input
    test_state = {
        "user_query": "Tell me a joke",
        "updated_query": "Tell me a joke",
    }
    
    print("\n🚀 Executing supervised agent...")
    result = await agent(test_state)
    
    print(f"\n📤 Result: {result}")
    print(f"\n📊 Agent Stats:")
    print(f"   - Executions: {agent.execution_count}")
    print(f"   - Interventions: {agent.intervention_count}")
    
    return result


if __name__ == "__main__":
    print("MAF Supervised Agents Pilot")
    print("=" * 60)
    
    # Try simple demo first (doesn't require full MAF setup)
    print("\nRunning standalone demo...")
    try:
        asyncio.run(demo_simple_supervised_intent())
    except Exception as e:
        print(f"\n⚠️ Note: Standalone demo requires MAF agents.")
        print(f"Error: {e}")
    
    # Try orchestrator demo
    print("\n" + "=" * 60)
    print("Attempting orchestrator demo (requires full MAF setup)...")
    try:
        asyncio.run(demo_supervised_orchestrator())
    except ImportError as e:
        print(f"⚠️ Orchestrator demo skipped (MAF components not available): {e}")
    except Exception as e:
        print(f"⚠️ Orchestrator demo error: {e}")
