"""
Phase 2: Expand Supervision to All MAF Subagents

After pilot success with IntentUnderstandingAgent, expand to all agents.
"""

import sys
from pathlib import Path

# Add workspace root to path for imports
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

from typing import Any, Dict, Optional
from experiment.agent_supervision import CorrectionType
from MAF.supervised_subagents import SubagentGovernor


# ==========================================
# Specialized Governors for Each Agent Type
# ==========================================

class GovernorActionExtractor(SubagentGovernor):
    """Governor for ActionExtractorAgent - validates action quality."""
    
    def _apply_subagent_rules(self, snapshot: Dict[str, Any]):
        """Rules for action extraction quality."""
        metrics = snapshot.get("metrics", {})
        quality = metrics.get("quality_score", 1.0)
        latency = metrics.get("latency_ms", 0)
        
        # RULE 1: No actions extracted
        if quality < 0.3:
            print(f"👑 [Action] RULE 1: No actions extracted")
            self._request_intervention(
                CorrectionType.RETRY,
                "Failed to extract any actions",
                priority=4
            )
        
        # RULE 2: Extraction too slow (LLM hallucination?)
        if latency > 4000:
            print(f"👑 [Action] RULE 2: Slow extraction ({latency:.0f}ms)")
            self._request_intervention(
                CorrectionType.MODIFY_PARAMS,
                "Action extraction taking too long",
                parameters={"simplify_prompt": True},
                priority=2
            )


class GovernorOrchestratorAgent(SubagentGovernor):
    """Governor for OrchestratorAgent - validates orchestration decisions."""
    
    def _apply_subagent_rules(self, snapshot: Dict[str, Any]):
        """Rules for orchestrator decision quality."""
        metrics = snapshot.get("metrics", {})
        quality = metrics.get("quality_score", 1.0)
        
        # RULE 1: No decision made
        if quality < 0.4:
            print(f"👑 [Orchestrator] RULE 1: Unclear decision")
            self._request_intervention(
                CorrectionType.ESCALATE,
                "Orchestrator made unclear decision",
                priority=5
            )


class GovernorPlannerAgent(SubagentGovernor):
    """Governor for PlannerAgent - validates plan quality."""
    
    def _apply_subagent_rules(self, snapshot: Dict[str, Any]):
        """Rules for plan quality."""
        metrics = snapshot.get("metrics", {})
        quality = metrics.get("quality_score", 1.0)
        
        # RULE 1: Plan too short or incomplete
        if quality < 0.5:
            print(f"👑 [Planner] RULE 1: Incomplete plan")
            self._request_intervention(
                CorrectionType.MODIFY_PARAMS,
                "Plan incomplete or too short",
                parameters={"min_steps": 3},
                priority=3
            )


class GovernorSummarizerAgent(SubagentGovernor):
    """Governor for SummarizerAgent - validates response quality."""
    
    def _apply_subagent_rules(self, snapshot: Dict[str, Any]):
        """Rules for summary quality."""
        metrics = snapshot.get("metrics", {})
        quality = metrics.get("quality_score", 1.0)
        
        # RULE 1: Empty or very short summary
        if quality < 0.3:
            print(f"👑 [Summarizer] RULE 1: Summary too short")
            self._request_intervention(
                CorrectionType.RETRY,
                "Summary is too short or empty",
                priority=3
            )


# ==========================================
# Supervised Wrappers for Each Agent
# ==========================================

class SupervisedActionExtractorAgent:
    """Wraps ActionExtractorAgent with supervision."""
    
    def __init__(self, original_agent: Any = None):
        self.original_agent = original_agent
        self.name = f"Supervised-ActionExtractor"
        self.execution_count = 0
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with governance."""
        from experiment.agent_supervision import SupervisedAgentState, AgentStatus
        from MAF.supervised_subagents import SubagentObserver
        import time
        
        self.execution_count += 1
        
        state_obj = SupervisedAgentState()
        state_obj.update("status", AgentStatus.WORKING)
        state_obj.update("current_operation", "Action Extraction")
        
        observer = SubagentObserver(state_obj, "action")
        governor = GovernorActionExtractor(state_obj, "action")
        
        observer.start()
        governor.start()
        
        try:
            start_time = time.time()
            result = await self.original_agent(state)
            latency_ms = (time.time() - start_time) * 1000
            
            quality = 0.9 if result.get("action_extraction", {}).get("actions") else 0.3
            state_obj.add_metric("latency_ms", latency_ms)
            state_obj.add_metric("quality_score", quality)
            
            state_obj.update("status", AgentStatus.COMPLETE)
            return result
        
        except Exception as e:
            state_obj.update("status", AgentStatus.ERROR)
            raise
        
        finally:
            observer.stop()
            governor.stop()
            observer.join(timeout=1)
            governor.join(timeout=1)


class SupervisedOrchestratorAgent:
    """Wraps OrchestratorAgent with supervision."""
    
    def __init__(self, original_agent: Any = None):
        self.original_agent = original_agent
        self.name = f"Supervised-Orchestrator"
        self.execution_count = 0
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with governance."""
        from experiment.agent_supervision import SupervisedAgentState, AgentStatus
        from MAF.supervised_subagents import SubagentObserver
        import time
        
        self.execution_count += 1
        
        state_obj = SupervisedAgentState()
        state_obj.update("status", AgentStatus.WORKING)
        state_obj.update("current_operation", "Orchestration")
        
        observer = SubagentObserver(state_obj, "orchestrator")
        governor = GovernorOrchestratorAgent(state_obj, "orchestrator")
        
        observer.start()
        governor.start()
        
        try:
            start_time = time.time()
            result = await self.original_agent(state)
            latency_ms = (time.time() - start_time) * 1000
            
            quality = 0.8 if result else 0.2
            state_obj.add_metric("latency_ms", latency_ms)
            state_obj.add_metric("quality_score", quality)
            
            state_obj.update("status", AgentStatus.COMPLETE)
            return result
        
        except Exception as e:
            state_obj.update("status", AgentStatus.ERROR)
            raise
        
        finally:
            observer.stop()
            governor.stop()
            observer.join(timeout=1)
            governor.join(timeout=1)


# ==========================================
# Factory Functions
# ==========================================

def create_supervised_agents(orchestrator: Any) -> None:
    """
    Upgrade all agents in orchestrator to supervised versions.
    
    Usage:
        from global_.orchestrator_graph import Global_Orchestrator
        from MAF.supervised_all_agents import create_supervised_agents
        
        orchestrator = Global_Orchestrator()
        create_supervised_agents(orchestrator)  # All agents now supervised
    """
    
    # Wrap all agents
    agents_to_wrap = [
        ("intent_understanding", "IntentUnderstandingAgent"),
        ("action_extractor", "ActionExtractorAgent"),
        ("orchestrator_agent", "OrchestratorAgent"),
    ]
    
    print("\n📦 Upgrading agents to supervised versions...")
    
    for attr_name, readable_name in agents_to_wrap:
        if hasattr(orchestrator, attr_name):
            agent = getattr(orchestrator, attr_name)
            
            # Import and wrap
            if "IntentUnderstanding" in readable_name:
                from MAF.supervised_agents_pilot import SupervisedIntentUnderstandingAgent
                wrapped = SupervisedIntentUnderstandingAgent(agent)
            elif "ActionExtractor" in readable_name:
                wrapped = SupervisedActionExtractorAgent(agent)
            elif "Orchestrator" in readable_name:
                wrapped = SupervisedOrchestratorAgent(agent)
            else:
                print(f"   ⚠️ {readable_name}: No wrapper available yet")
                continue
            
            setattr(orchestrator, attr_name, wrapped)
            print(f"   ✓ {readable_name}: Supervised")


# ==========================================
# Monitoring Dashboard
# ==========================================

def get_supervision_metrics(orchestrator: Any) -> Dict[str, Any]:
    """
    Get supervision metrics from all supervised agents.
    
    Usage:
        metrics = get_supervision_metrics(orchestrator)
        print(metrics)
    """
    
    metrics = {}
    
    agents = [
        ("intent_understanding", "Intent"),
        ("action_extractor", "Action"),
        ("orchestrator_agent", "Orchestrator"),
    ]
    
    for attr_name, readable_name in agents:
        if hasattr(orchestrator, attr_name):
            agent = getattr(orchestrator, attr_name)
            
            # Check if it's supervised
            if hasattr(agent, 'execution_count'):
                metrics[readable_name] = {
                    "executions": agent.execution_count,
                    "interventions": agent.intervention_count if hasattr(agent, 'intervention_count') else 0,
                }
    
    return metrics


# ==========================================
# Configuration & Customization
# ==========================================

class SupervisionConfig:
    """Configuration for agent supervision."""
    
    # Intent Agent thresholds
    INTENT_QUALITY_THRESHOLD = 0.7
    INTENT_LATENCY_THRESHOLD = 3000  # ms
    
    # Action Agent thresholds
    ACTION_QUALITY_THRESHOLD = 0.5
    ACTION_LATENCY_THRESHOLD = 4000  # ms
    
    # Orchestrator thresholds
    ORCHESTRATOR_QUALITY_THRESHOLD = 0.6
    ORCHESTRATOR_LATENCY_THRESHOLD = 5000  # ms
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Get config as dictionary."""
        return {
            "intent": {
                "quality_threshold": cls.INTENT_QUALITY_THRESHOLD,
                "latency_threshold": cls.INTENT_LATENCY_THRESHOLD,
            },
            "action": {
                "quality_threshold": cls.ACTION_QUALITY_THRESHOLD,
                "latency_threshold": cls.ACTION_LATENCY_THRESHOLD,
            },
            "orchestrator": {
                "quality_threshold": cls.ORCHESTRATOR_QUALITY_THRESHOLD,
                "latency_threshold": cls.ORCHESTRATOR_LATENCY_THRESHOLD,
            },
        }


if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "=" * 70)
    print("Phase 2: Expand Supervision to All Agents")
    print("=" * 70)
    
    print("""
    After Phase 1 success with IntentUnderstandingAgent, Phase 2 expands
    supervision to all MAF subagents.
    
    NEW COMPONENTS:
    - GovernorActionExtractor: Rules for action extraction quality
    - GovernorOrchestratorAgent: Rules for orchestration decisions
    - GovernorPlannerAgent: Rules for plan quality
    - GovernorSummarizerAgent: Rules for response quality
    
    - SupervisedActionExtractorAgent: Wrapper for action agent
    - SupervisedOrchestratorAgent: Wrapper for orchestrator
    
    QUICK START:
    
    from global_.orchestrator_graph import Global_Orchestrator
    from MAF.supervised_all_agents import create_supervised_agents
    
    orchestrator = Global_Orchestrator()
    create_supervised_agents(orchestrator)  # All agents supervised!
    
    # Use orchestrator normally
    result = await orchestrator.somemethod(state)
    
    NEXT STEPS:
    1. Test Phase 1 with real MAF workflow
    2. Implement wrappers for remaining agents
    3. Add custom governance rules per domain
    4. Deploy with feature flags
    """)
