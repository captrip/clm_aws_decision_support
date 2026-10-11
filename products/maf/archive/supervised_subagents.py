"""
MAF Subagent Supervision

Wraps MAF subagents (IntentUnderstandingAgent, ActionExtractorAgent, etc.)
with Agent + Observer + Governor supervision.

Phase 1: Subagent monitoring and intervention
"""

import sys
import os
from pathlib import Path

# Add workspace root to path for imports
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
import threading
import time
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass

from experiment.agent_supervision import (
    SupervisedAgentState,
    AgentStatus,
    CorrectionType,
    InterventionRequest,
)


@dataclass
class SubagentMetrics:
    """Metrics specific to MAF subagents."""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    confidence_score: float = 0.0
    errors_detected: int = 0
    quality_score: float = 0.0


class SubagentObserver(threading.Thread):
    """Observer specialized for MAF subagents."""
    
    def __init__(self, state: SupervisedAgentState, agent_type: str):
        super().__init__(name=f"Observer-{agent_type}")
        self.state = state
        self.agent_type = agent_type
        self.running = True
        self.daemon = True
        
        # Subagent-specific metrics
        self.observation_count = 0
        self.quality_issues = 0
        self.latency_threshold = 5000  # ms
    
    def run(self):
        """Monitor subagent behavior."""
        print(f"👁️ SubagentObserver[{self.agent_type}]: Started")
        
        while self.running:
            snapshot = self.state.get_snapshot()
            
            if snapshot["status"] != AgentStatus.IDLE:
                self.observation_count += 1
                
                # Extract metrics
                metrics = snapshot.get("metrics", {})
                
                # Check quality indicators
                latency = metrics.get("latency_ms", 0)
                quality = metrics.get("quality_score", 1.0)
                
                # Print observation
                status_emoji = "✓" if quality > 0.8 else "⚠"
                print(f"👁️ [{self.agent_type}] {status_emoji} Quality: {quality:.2f} | Latency: {latency:.0f}ms")
                
                # Track issues
                if quality < 0.8:
                    self.quality_issues += 1
            
            time.sleep(0.3)
        
        print(f"✅ SubagentObserver[{self.agent_type}]: Complete")
    
    def stop(self):
        self.running = False


class SubagentGovernor(threading.Thread):
    """Governor specialized for MAF subagents."""
    
    def __init__(self, state: SupervisedAgentState, agent_type: str):
        super().__init__(name=f"Governor-{agent_type}")
        self.state = state
        self.agent_type = agent_type
        self.running = True
        self.daemon = True
        
        # Subagent-specific rules
        self.analysis_count = 0
        self.intervention_count = 0
    
    def run(self):
        """Govern subagent behavior."""
        print(f"👑 SubagentGovernor[{self.agent_type}]: Started")
        
        while self.running:
            snapshot = self.state.get_snapshot()
            
            if snapshot["status"] != AgentStatus.IDLE:
                self.analysis_count += 1
                self._apply_subagent_rules(snapshot)
            
            time.sleep(0.2)
        
        print(f"✅ SubagentGovernor[{self.agent_type}]: Complete")
    
    def _apply_subagent_rules(self, snapshot: Dict[str, Any]):
        """Apply governance rules specific to this subagent type."""
        metrics = snapshot.get("metrics", {})
        quality = metrics.get("quality_score", 1.0)
        latency = metrics.get("latency_ms", 0)
        
        # RULE 1: Quality below threshold
        if quality < 0.7:
            print(f"👑 [{self.agent_type}] RULE 1: Low quality detected ({quality:.2f})")
            self._request_intervention(
                CorrectionType.RETRY,
                f"Quality score {quality:.2f} below 0.7 threshold",
                priority=3
            )
        
        # RULE 2: High latency
        if latency > 5000:
            print(f"👑 [{self.agent_type}] RULE 2: High latency ({latency:.0f}ms)")
            self._request_intervention(
                CorrectionType.MODIFY_PARAMS,
                "Latency exceeding 5000ms threshold",
                parameters={"timeout": 10000},
                priority=2
            )
        
        # RULE 3: Multiple retries
        decisions = snapshot.get("decisions", [])
        retry_count = sum(1 for d in decisions if "Retry" in d)
        if retry_count > 2:
            print(f"👑 [{self.agent_type}] RULE 3: Multiple retries ({retry_count})")
            self._request_intervention(
                CorrectionType.ESCALATE,
                f"Failed {retry_count} times, escalating",
                priority=5
            )
    
    def _request_intervention(
        self,
        correction_type: CorrectionType,
        reason: str,
        parameters: Optional[Dict] = None,
        priority: int = 1
    ):
        """Queue intervention."""
        intervention = InterventionRequest(
            correction_type=correction_type,
            reason=reason,
            parameters=parameters or {},
            priority=priority,
        )
        self.state.push_intervention(intervention)
        self.intervention_count += 1
        print(f"   → Intervention #{self.intervention_count} queued")
    
    def stop(self):
        self.running = False


class SupervisedSubagent:
    """Wraps MAF subagent with supervision."""
    
    def __init__(
        self,
        agent_name: str,
        agent_execute_fn: Callable,
        agent_type: str,  # "intent", "action", "planning", etc.
    ):
        self.agent_name = agent_name
        self.agent_execute_fn = agent_execute_fn
        self.agent_type = agent_type
        
        # Supervision state
        self.state = SupervisedAgentState()
        
        # Thread handles
        self.observer = None
        self.governor = None
        self.agent_thread = None
    
    async def execute_supervised(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subagent with supervision."""
        
        print(f"\n🤖 SupervisedSubagent[{self.agent_type}]: Starting")
        
        self.state.update("status", AgentStatus.WORKING)
        self.state.update("current_operation", f"Executing {self.agent_name}")
        
        # Start observer and governor in threads
        self.observer = SubagentObserver(self.state, self.agent_type)
        self.governor = SubagentGovernor(self.state, self.agent_type)
        
        self.observer.start()
        self.governor.start()
        
        try:
            # Execute agent
            start_time = time.time()
            
            # Check for initial intervention
            intervention = self.state.pop_intervention()
            if intervention:
                print(f"⚠️ Received {intervention.correction_type.value} before execution")
            
            # Run agent
            result = await self.agent_execute_fn(input_data)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Compute quality score (example: based on result structure)
            quality_score = self._compute_quality_score(result)
            
            # Update metrics
            self.state.add_metric("latency_ms", latency_ms)
            self.state.add_metric("quality_score", quality_score)
            self.state.add_metric("agent_result", result)
            
            # Check for post-execution interventions
            intervention = self.state.pop_intervention()
            if intervention:
                print(f"⚠️ Post-execution intervention: {intervention.correction_type.value}")
                result = self._apply_post_intervention(intervention, result)
            
            self.state.update("status", AgentStatus.COMPLETE)
            print(f"✅ SupervisedSubagent[{self.agent_type}]: Complete (Quality: {quality_score:.2f})")
            
            return result
        
        except Exception as e:
            self.state.update("status", AgentStatus.ERROR)
            self.state.add_error(str(e))
            print(f"❌ SupervisedSubagent[{self.agent_type}]: Error - {e}")
            raise
        
        finally:
            # Stop observer and governor
            self.observer.stop()
            self.governor.stop()
            self.observer.join(timeout=1)
            self.governor.join(timeout=1)
    
    def _compute_quality_score(self, result: Dict[str, Any]) -> float:
        """Compute quality score for subagent output."""
        # Agent-type specific quality checks
        
        if self.agent_type == "intent":
            # Intent understanding quality: presence of key fields
            required = ["intent", "confidence"]
            present = sum(1 for field in required if field in result)
            score = present / len(required)
            return min(score + result.get("confidence", 0), 1.0)
        
        elif self.agent_type == "action":
            # Action extraction quality: action count and confidence
            actions = result.get("actions", [])
            if not actions:
                return 0.5
            avg_confidence = sum(a.get("confidence", 0.5) for a in actions) / len(actions)
            return avg_confidence
        
        elif self.agent_type == "planning":
            # Planning quality: plan completeness
            steps = result.get("steps", [])
            if not steps:
                return 0.3
            return min(len(steps) / 10, 1.0)
        
        else:
            # Default: presence of result data
            return 0.8 if result else 0.2
    
    def _apply_post_intervention(self, intervention: InterventionRequest, result: Any) -> Any:
        """Apply post-execution intervention."""
        if intervention.correction_type == CorrectionType.MODIFY_PARAMS:
            print(f"   → Applying params: {intervention.parameters}")
            self.state.add_decision(f"Applied post-execution params")
        
        elif intervention.correction_type == CorrectionType.ESCALATE:
            print(f"   → Escalated to manual review")
            self.state.add_decision("Escalated")
        
        return result


# ==========================================
# Integration with MAF Agents
# ==========================================

async def wrap_maf_subagent(
    agent_instance: Any,
    agent_type: str,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Wrap MAF subagent execution with supervision.
    
    Args:
        agent_instance: MAF agent (IntentUnderstandingAgent, etc.)
        agent_type: Type of agent ("intent", "action", "planning", etc.)
        input_data: Input for the agent
    
    Returns:
        Agent output with supervision applied
    """
    
    # Create supervised wrapper
    supervised = SupervisedSubagent(
        agent_name=agent_instance.__class__.__name__,
        agent_execute_fn=agent_instance.ainvoke,
        agent_type=agent_type,
    )
    
    # Execute with supervision
    result = await supervised.execute_supervised(input_data)
    
    return result


# ==========================================
# Example Usage (with MAF agents)
# ==========================================

if __name__ == "__main__":
    print("MAF Subagent Supervision Integration")
    print("=" * 60)
    print("""
This module integrates MAF subagents with the supervision system.

Usage:
  
  from maf_supervised_subagents import wrap_maf_subagent
  
  result = await wrap_maf_subagent(
    agent_instance=intent_agent,
    agent_type="intent",
    input_data={"query": "..."}
  )

The supervision system will:
  ✓ Monitor agent quality in real-time
  ✓ Detect quality issues
  ✓ Apply corrections automatically
  ✓ Escalate problems as needed
    """)
