"""
Advanced Integration: Supervised MAF Agent

Shows how to integrate the SupervisedAgent system with actual MAF agents
to get real-time monitoring and intervention capabilities.
"""

import asyncio
import threading
import time
from typing import Any, Dict, Optional, Callable
from agent_supervision import (
    SupervisedAgent,
    SupervisedAgentState,
    AgentThread,
    ObserverThread,
    GovernorThread,
    CorrectionType,
    AgentStatus,
)


# ==========================================
# 1. MAF Agent Wrapper for Supervision
# ==========================================

class SupervisedMAFAgent:
    """
    Wraps a MAF agent with supervision capabilities.
    
    The agent runs in its own thread, while observer and governor
    monitor and correct its behavior in parallel.
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_execute_fn: Callable,
        state_manager: Optional[Any] = None,
    ):
        """
        Args:
            agent_name: Name of the MAF agent
            agent_execute_fn: Async function that executes the agent
            state_manager: MAF StateManager instance (optional)
        """
        self.agent_name = agent_name
        self.agent_execute_fn = agent_execute_fn
        self.state_manager = state_manager
        
        # Supervision state
        self.supervision_state = SupervisedAgentState()
        
        # Thread objects
        self.agent_thread_obj = None
        self.observer_thread_obj = None
        self.governor_thread_obj = None
        
        # Control
        self.running = True
    
    def _agent_wrapper(self):
        """
        Wraps MAF agent execution in supervised thread.
        
        This runs in a thread and periodically checks for governor corrections.
        """
        print(f"🤖 MAF Agent '{self.agent_name}': Started")
        
        self.supervision_state.update("status", AgentStatus.WORKING)
        self.supervision_state.update("start_time", time.time())
        
        try:
            # Run the async agent in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Update state before execution
            self.supervision_state.update("current_operation", f"Executing {self.agent_name}")
            
            # Execute agent
            result = loop.run_until_complete(self.agent_execute_fn())
            
            # Check for interventions after execution
            intervention = self.supervision_state.pop_intervention()
            if intervention:
                print(f"⚠️ MAF Agent: Intervention received - {intervention.correction_type.value}")
                self._handle_maf_intervention(intervention, result)
            
            # Store result
            self.supervision_state.update("status", AgentStatus.COMPLETE)
            self.supervision_state.add_metric("agent_result", result)
            
            # Sync with state manager if available
            if self.state_manager:
                self.state_manager.update(
                    f"supervised_{self.agent_name}",
                    {
                        "status": "complete",
                        "result": result,
                        "supervised": True,
                    }
                )
            
            print(f"✅ MAF Agent '{self.agent_name}': Complete")
        
        except Exception as e:
            self.supervision_state.update("status", AgentStatus.ERROR)
            self.supervision_state.add_error(f"MAF agent failed: {str(e)}")
            print(f"❌ MAF Agent '{self.agent_name}': Error - {e}")
    
    def _observer_for_maf(self):
        """Observer specialized for MAF agents."""
        print(f"👁️ Observer for '{self.agent_name}': Started")
        
        observation_count = 0
        while self.running:
            snapshot = self.supervision_state.get_snapshot()
            observation_count += 1
            
            if snapshot["status"] != AgentStatus.IDLE:
                print(f"👁️ Observer: [{observation_count}] {self.agent_name} -> {snapshot['status'].value}")
                print(f"   Operation: {snapshot['current_operation']}")
                print(f"   Metrics: {snapshot['metrics']}")
            
            time.sleep(0.5)
        
        print(f"✅ Observer for '{self.agent_name}': Complete")
    
    def _governor_for_maf(self):
        """Governor specialized for MAF agents."""
        print(f"👑 Governor for '{self.agent_name}': Started")
        
        analysis_count = 0
        while self.running:
            snapshot = self.supervision_state.get_snapshot()
            analysis_count += 1
            
            # Example governance rules for MAF agents
            
            # RULE: If agent is stuck in same operation too long
            if (snapshot["current_operation"] and 
                time.time() - self.supervision_state.get("start_time", time.time()) > 5):
                
                print(f"👑 Governor: ⚠️ {self.agent_name} execution taking long time")
                
                # Request retry or modification
                from agent_supervision import InterventionRequest
                intervention = InterventionRequest(
                    correction_type=CorrectionType.MODIFY_PARAMS,
                    reason="Agent execution timeout",
                    parameters={"timeout_multiplier": 2.0},
                    priority=3,
                )
                self.supervision_state.push_intervention(intervention)
            
            # RULE: If errors occurred, escalate
            errors = snapshot["errors"]
            if len(errors) > 0:
                print(f"👑 Governor: 🚨 {self.agent_name} has errors - escalating")
                
                from agent_supervision import InterventionRequest
                intervention = InterventionRequest(
                    correction_type=CorrectionType.ESCALATE,
                    reason=f"Errors: {errors}",
                    priority=5,
                )
                self.supervision_state.push_intervention(intervention)
            
            time.sleep(0.3)
        
        print(f"✅ Governor for '{self.agent_name}': Complete")
    
    def _handle_maf_intervention(self, intervention, agent_result: Any):
        """Handle intervention for MAF agent."""
        correction_type = intervention.correction_type
        reason = intervention.reason
        params = intervention.parameters
        
        print(f"⚠️ MAF Agent Intervention: {correction_type.value}")
        print(f"   Reason: {reason}")
        print(f"   Params: {params}")
        
        # Apply specific interventions
        if correction_type == CorrectionType.MODIFY_PARAMS:
            print(f"   → Modifying agent parameters: {params}")
            self.supervision_state.add_decision(f"Applied params: {params}")
        
        elif correction_type == CorrectionType.RETRY:
            print(f"   → Retrying agent execution")
            self.supervision_state.add_decision("Retry executed")
        
        elif correction_type == CorrectionType.ESCALATE:
            print(f"   → Escalating to human review")
            self.supervision_state.add_decision("Escalated to human")
    
    async def run_supervised(self):
        """Run the MAF agent with supervision."""
        print("=" * 70)
        print(f"SUPERVISED MAF AGENT: {self.agent_name}")
        print("=" * 70 + "\n")
        
        # Create thread objects
        agent_thread_obj = AgentThread(
            state=self.supervision_state,
            work_function=lambda step: {"step": step, "status": "working"},
            total_steps=1,  # MAF agents run once
        )
        
        self.observer_thread_obj = ObserverThread(self.supervision_state)
        self.governor_thread_obj = GovernorThread(self.supervision_state)
        
        # Create threads
        agent_t = threading.Thread(target=self._agent_wrapper, name=f"MAF-{self.agent_name}")
        observer_t = threading.Thread(target=self._observer_for_maf, name=f"Observer-{self.agent_name}")
        governor_t = threading.Thread(target=self._governor_for_maf, name=f"Governor-{self.agent_name}")
        
        # Start threads
        agent_t.start()
        observer_t.start()
        governor_t.start()
        
        # Wait for agent to complete
        agent_t.join()
        
        # Stop observer and governor
        self.running = False
        observer_t.join()
        governor_t.join()
        
        # Return final state
        return self.supervision_state.get_snapshot()


# ==========================================
# 2. Multi-Agent Supervision
# ==========================================

class SupervisedMultiAgentSystem:
    """
    Manages supervision of multiple MAF agents running in parallel.
    
    Each agent has its own supervisor, but all share a global governor.
    """
    
    def __init__(self):
        self.agents: Dict[str, SupervisedMAFAgent] = {}
        self.global_supervision_state = SupervisedAgentState()
    
    def register_agent(
        self,
        agent_name: str,
        agent_execute_fn: Callable,
        state_manager: Optional[Any] = None,
    ):
        """Register an agent for supervision."""
        supervised_agent = SupervisedMAFAgent(
            agent_name=agent_name,
            agent_execute_fn=agent_execute_fn,
            state_manager=state_manager,
        )
        self.agents[agent_name] = supervised_agent
    
    async def run_all_supervised(self):
        """Run all agents with supervision."""
        print("=" * 70)
        print("MULTI-AGENT SUPERVISION SYSTEM")
        print("=" * 70 + "\n")
        
        # Run all agents concurrently
        tasks = [
            agent.run_supervised()
            for agent in self.agents.values()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        summary = {
            agent_name: results[i]
            for i, agent_name in enumerate(self.agents.keys())
        }
        
        print("\n" + "=" * 70)
        print("MULTI-AGENT EXECUTION SUMMARY")
        print("=" * 70)
        
        for agent_name, result in summary.items():
            print(f"\n{agent_name}:")
            print(f"  Status: {result['status'].value}")
            print(f"  Decisions: {len(result['decisions'])}")
            print(f"  Errors: {len(result['errors'])}")
        
        return summary


# ==========================================
# 3. Example Usage
# ==========================================

async def example_maf_agent_function():
    """
    Example async function that simulates a MAF agent.
    """
    print("   → Executing simulated MAF agent logic")
    await asyncio.sleep(1)  # Simulate work
    return {"agent_output": "Success", "processed_items": 42}


async def main():
    """Main example."""
    
    # Example 1: Single supervised MAF agent
    print("\n--- Example 1: Single Supervised MAF Agent ---\n")
    
    supervised = SupervisedMAFAgent(
        agent_name="WebSearchAgent",
        agent_execute_fn=example_maf_agent_function,
    )
    
    result = await supervised.run_supervised()
    print(f"\nFinal State: {result['status'].value}")
    
    # Example 2: Multiple supervised MAF agents
    print("\n\n--- Example 2: Multi-Agent Supervision ---\n")
    
    multi_system = SupervisedMultiAgentSystem()
    
    multi_system.register_agent(
        "SearchAgent",
        example_maf_agent_function,
    )
    multi_system.register_agent(
        "AnalysisAgent",
        example_maf_agent_function,
    )
    multi_system.register_agent(
        "SummaryAgent",
        example_maf_agent_function,
    )
    
    results = await multi_system.run_all_supervised()


if __name__ == "__main__":
    asyncio.run(main())
