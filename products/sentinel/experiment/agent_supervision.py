"""
Agent + Observer + Governor System

A true parallel supervision framework where:
- Agent Thread: Executes main user work
- Observer Thread: Watches agent behavior in real-time
- Governor Thread: Analyzes and makes intervention decisions

All three run concurrently with thread-safe state management.
"""

import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List
from enum import Enum
from uuid import uuid4


# ==========================================
# 1. Enums and Data Structures
# ==========================================

class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    CORRECTING = "correcting"
    HALTED = "halted"
    COMPLETE = "complete"
    ERROR = "error"


class CorrectionType(Enum):
    """Types of interventions governor can apply."""
    RETRY = "retry"
    SKIP = "skip"
    MODIFY_PARAMS = "modify_params"
    PAUSE = "pause"
    RESUME = "resume"
    HALT = "halt"
    ESCALATE = "escalate"


@dataclass
class ObservationSnapshot:
    """Snapshot of agent state at a point in time."""
    timestamp: float
    status: AgentStatus
    current_operation: str
    progress: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    decisions: List[str] = field(default_factory=list)


@dataclass
class InterventionRequest:
    """Request from Governor to Agent."""
    correction_type: CorrectionType
    reason: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 5=critical


# ==========================================
# 2. Thread-Safe State Management
# ==========================================

class SupervisedAgentState:
    """Thread-safe state container for supervised agent."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "status": AgentStatus.IDLE,
            "current_operation": "",
            "progress": 0,
            "total_steps": 0,
            "start_time": None,
            "metrics": {},
            "decisions": [],
            "errors": [],
            "observation_queue": [],
            "intervention_queue": [],
            "execution_id": str(uuid4()),
        }
    
    def update(self, key: str, value: Any) -> None:
        """Thread-safe state update."""
        with self._lock:
            self._state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe state read."""
        with self._lock:
            return self._state.get(key, default)
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get complete state snapshot (thread-safe copy)."""
        with self._lock:
            return {
                "status": self._state["status"],
                "current_operation": self._state["current_operation"],
                "progress": self._state["progress"],
                "total_steps": self._state["total_steps"],
                "metrics": self._state["metrics"].copy(),
                "decisions": self._state["decisions"].copy(),
                "errors": self._state["errors"].copy(),
                "execution_id": self._state["execution_id"],
            }
    
    def push_observation(self, observation: ObservationSnapshot) -> None:
        """Push observation to queue."""
        with self._lock:
            self._state["observation_queue"].append(observation)
    
    def pop_observation(self) -> Optional[ObservationSnapshot]:
        """Pop oldest observation."""
        with self._lock:
            if self._state["observation_queue"]:
                return self._state["observation_queue"].pop(0)
            return None
    
    def push_intervention(self, intervention: InterventionRequest) -> None:
        """Push intervention request."""
        with self._lock:
            self._state["intervention_queue"].append(intervention)
    
    def pop_intervention(self) -> Optional[InterventionRequest]:
        """Pop highest priority intervention."""
        with self._lock:
            if not self._state["intervention_queue"]:
                return None
            # Sort by priority (highest first) and pop max
            self._state["intervention_queue"].sort(
                key=lambda x: x.priority, reverse=True
            )
            return self._state["intervention_queue"].pop(0)
    
    def has_pending_interventions(self) -> bool:
        """Check if interventions are waiting."""
        with self._lock:
            return len(self._state["intervention_queue"]) > 0
    
    def add_metric(self, key: str, value: Any) -> None:
        """Add/update metric."""
        with self._lock:
            self._state["metrics"][key] = value
    
    def add_decision(self, decision: str) -> None:
        """Log a decision."""
        with self._lock:
            self._state["decisions"].append(decision)
    
    def add_error(self, error: str) -> None:
        """Log an error."""
        with self._lock:
            self._state["errors"].append(error)


# ==========================================
# 3. Agent Thread (Main Work)
# ==========================================

class AgentThread:
    """
    Main agent thread that executes user work.
    
    Periodically checks for interventions from governor
    and applies corrections as needed.
    """
    
    def __init__(
        self,
        state: SupervisedAgentState,
        work_function: Callable,
        total_steps: int = 10,
    ):
        self.state = state
        self.work_function = work_function
        self.total_steps = total_steps
        self.current_step = 0
        self.running = True
    
    def run(self):
        """Execute agent work."""
        print("🤖 Agent Thread: Started")
        
        self.state.update("status", AgentStatus.WORKING)
        self.state.update("total_steps", self.total_steps)
        self.state.update("start_time", time.time())
        
        try:
            for step in range(1, self.total_steps + 1):
                if not self.running:
                    break
                
                self.current_step = step
                
                # CHECK FOR PENDING INTERVENTIONS
                intervention = self.state.pop_intervention()
                if intervention:
                    await_correction = self._handle_intervention(intervention)
                    if not await_correction:
                        continue
                
                # Execute work
                operation_name = f"Step {step}"
                self.state.update("current_operation", operation_name)
                self.state.update("progress", step)
                
                print(f"📋 Agent: {operation_name}")
                
                # Call user work function
                result = self.work_function(step)
                
                # Update metrics
                self.state.add_metric(f"step_{step}_result", result)
                self.state.add_metric(f"step_{step}_time", time.time())
                
                # Small delay to allow other threads to run
                time.sleep(0.2)
            
            self.state.update("status", AgentStatus.COMPLETE)
            print("✅ Agent Thread: Complete")
        
        except Exception as e:
            self.state.update("status", AgentStatus.ERROR)
            self.state.add_error(f"Agent execution failed: {str(e)}")
            print(f"❌ Agent Thread: Error - {e}")
    
    def _handle_intervention(self, intervention: InterventionRequest) -> bool:
        """
        Handle intervention from governor.
        
        Returns:
            True if agent should continue normally
            False if agent should skip to next iteration
        """
        print(f"\n⚠️ Agent: Received {intervention.correction_type.value} intervention")
        print(f"   Reason: {intervention.reason}")
        
        if intervention.correction_type == CorrectionType.RETRY:
            print(f"   → Retrying current step")
            self.state.add_decision(f"Retry step {self.current_step}")
            return True  # Retry same step
        
        elif intervention.correction_type == CorrectionType.SKIP:
            print(f"   → Skipping to next step")
            self.state.add_decision(f"Skip step {self.current_step}")
            return False  # Skip this step
        
        elif intervention.correction_type == CorrectionType.MODIFY_PARAMS:
            params = intervention.parameters
            print(f"   → Modifying parameters: {params}")
            self.state.add_decision(f"Modified params: {params}")
            self.state.add_metric("param_modifications", params)
            return True
        
        elif intervention.correction_type == CorrectionType.PAUSE:
            print(f"   → Pausing execution")
            self.state.update("status", AgentStatus.PAUSED)
            self.state.add_decision("Paused by governor")
            time.sleep(1)  # Simulate pause
            self.state.update("status", AgentStatus.WORKING)
            return True
        
        elif intervention.correction_type == CorrectionType.HALT:
            print(f"   → HALTING execution")
            self.state.update("status", AgentStatus.HALTED)
            self.state.add_decision("Halted by governor")
            self.running = False
            return False
        
        elif intervention.correction_type == CorrectionType.ESCALATE:
            print(f"   → ESCALATING (requires manual review)")
            self.state.update("status", AgentStatus.PAUSED)
            self.state.add_decision("Escalated to manual review")
            return True
        
        return True


# ==========================================
# 4. Observer Thread (Real-time Monitoring)
# ==========================================

class ObserverThread:
    """
    Observer thread that watches agent behavior in real-time.
    
    Continuously monitors agent state and publishes observations
    for governor analysis.
    """
    
    def __init__(self, state: SupervisedAgentState):
        self.state = state
        self.running = True
        self.observation_count = 0
    
    def run(self):
        """Continuously observe agent."""
        print("👁️ Observer Thread: Started")
        
        while self.running:
            # Get current state
            snapshot = self._create_snapshot()
            
            # Publish observation
            self.state.push_observation(snapshot)
            self.observation_count += 1
            
            # Display observation
            status = snapshot.status.value
            operation = snapshot.current_operation
            progress_pct = (snapshot.progress / snapshot.progress) * 100 if snapshot.progress > 0 else 0
            
            if operation:
                print(f"👁️ Observer: [{self.observation_count}] {status:10} | {operation:20} | {snapshot.progress}/{snapshot.progress} steps")
            
            # Wait before next observation
            time.sleep(0.5)
        
        print("✅ Observer Thread: Complete")
    
    def _create_snapshot(self) -> ObservationSnapshot:
        """Create observation snapshot."""
        state_dict = self.state.get_snapshot()
        
        return ObservationSnapshot(
            timestamp=time.time(),
            status=state_dict["status"],
            current_operation=state_dict["current_operation"],
            progress=state_dict["progress"],
            metrics=state_dict["metrics"],
            decisions=state_dict["decisions"],
        )
    
    def stop(self):
        """Stop observer."""
        self.running = False


# ==========================================
# 5. Governor Thread (Decision Making & Intervention)
# ==========================================

class GovernorThread:
    """
    Governor thread that analyzes agent behavior and applies interventions.
    
    Uses observations from observer to make decisions about:
    - Whether agent is progressing correctly
    - If corrections are needed
    - What interventions to apply
    """
    
    def __init__(self, state: SupervisedAgentState):
        self.state = state
        self.running = True
        self.analysis_count = 0
        self.intervention_count = 0
    
    def run(self):
        """Continuously analyze and govern."""
        print("👑 Governor Thread: Started")
        
        while self.running:
            # Get latest observations
            observation = self.state.pop_observation()
            
            if observation:
                self.analysis_count += 1
                self._analyze_and_intervene(observation)
            
            # Wait before next analysis
            time.sleep(0.3)
        
        print("✅ Governor Thread: Complete")
    
    def _analyze_and_intervene(self, observation: ObservationSnapshot) -> None:
        """Analyze observation and decide if intervention needed."""
        
        print(f"👑 Governor: [Analysis #{self.analysis_count}] Reviewing state...")
        
        # RULE 1: If step 2, it's taking a long time - request skip
        if observation.progress == 2 and observation.current_operation.endswith("Step 2"):
            print("👑 Governor: ⚠️ RULE 1 triggered - Step 2 taking too long")
            self._request_intervention(
                CorrectionType.SKIP,
                "Step 2 exceeded time threshold",
                priority=3
            )
        
        # RULE 2: If progress stalled (same step for too long)
        metrics = observation.metrics
        step_times = [v for k, v in metrics.items() if k.startswith("step_") and k.endswith("_time")]
        if len(step_times) > 2:
            print("👑 Governor: ✓ RULE 2 check - Progress is steady, no intervention")
        
        # RULE 3: If more than 7 steps, propose modification
        if observation.progress > 7:
            print("👑 Governor: ⚠️ RULE 3 triggered - Many steps executed")
            self._request_intervention(
                CorrectionType.MODIFY_PARAMS,
                "Agent has processed 7+ steps, optimizing parameters",
                parameters={"efficiency_boost": 1.5, "timeout": 30},
                priority=2
            )
        
        # RULE 4: If errors detected, escalate
        if observation.progress > 5 and len(self.state.get("errors", [])) > 0:
            print("👑 Governor: 🚨 RULE 4 triggered - Errors detected")
            self._request_intervention(
                CorrectionType.ESCALATE,
                "Errors detected during execution",
                priority=5
            )
    
    def _request_intervention(
        self,
        correction_type: CorrectionType,
        reason: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 1,
    ) -> None:
        """Queue an intervention request."""
        intervention = InterventionRequest(
            correction_type=correction_type,
            reason=reason,
            parameters=parameters or {},
            priority=priority,
        )
        
        self.state.push_intervention(intervention)
        self.intervention_count += 1
        
        print(f"   → Intervention #{self.intervention_count} queued: {correction_type.value}")
    
    def stop(self):
        """Stop governor."""
        self.running = False


# ==========================================
# 6. Main Orchestrator
# ==========================================

class SupervisedAgent:
    """
    Orchestrates all three threads (Agent, Observer, Governor).
    
    Entry point for creating and running a supervised agent system.
    """
    
    def __init__(self, work_function: Callable, total_steps: int = 10):
        """
        Args:
            work_function: Callable that takes step number and returns result
            total_steps: Total number of steps to execute
        """
        self.state = SupervisedAgentState()
        self.work_function = work_function
        self.total_steps = total_steps
        
        # Create thread instances
        self.agent_thread_obj = AgentThread(self.state, work_function, total_steps)
        self.observer_thread_obj = ObserverThread(self.state)
        self.governor_thread_obj = GovernorThread(self.state)
        
        # Thread handles
        self.agent_thread = None
        self.observer_thread = None
        self.governor_thread = None
    
    def run(self):
        """Start all three threads and wait for completion."""
        print("=" * 70)
        print("SUPERVISED AGENT SYSTEM (Agent + Observer + Governor)")
        print("=" * 70 + "\n")
        
        # Create threads
        self.agent_thread = threading.Thread(
            target=self.agent_thread_obj.run,
            name="Agent",
        )
        self.observer_thread = threading.Thread(
            target=self.observer_thread_obj.run,
            name="Observer",
        )
        self.governor_thread = threading.Thread(
            target=self.governor_thread_obj.run,
            name="Governor",
        )
        
        # Start all threads
        self.agent_thread.start()
        self.observer_thread.start()
        self.governor_thread.start()
        
        # Wait for agent to complete
        self.agent_thread.join()
        
        # Stop observer and governor
        self.observer_thread_obj.stop()
        self.governor_thread_obj.stop()
        
        self.observer_thread.join()
        self.governor_thread.join()
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print execution summary."""
        print("\n" + "=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        
        snapshot = self.state.get_snapshot()
        
        print(f"\nExecution ID: {snapshot['execution_id']}")
        print(f"Final Status: {snapshot['status'].value}")
        print(f"Progress: {snapshot['progress']}/{snapshot['total_steps']} steps")
        
        print(f"\nDecisions Made: {len(snapshot['decisions'])}")
        for i, decision in enumerate(snapshot['decisions'], 1):
            print(f"  {i}. {decision}")
        
        print(f"\nMetrics:")
        for key, value in snapshot['metrics'].items():
            print(f"  - {key}: {value}")
        
        if snapshot['errors']:
            print(f"\nErrors: {len(snapshot['errors'])}")
            for i, error in enumerate(snapshot['errors'], 1):
                print(f"  {i}. {error}")
        
        print(f"\nObserver Observations: {self.observer_thread_obj.observation_count}")
        print(f"Governor Analyses: {self.governor_thread_obj.analysis_count}")
        print(f"Governor Interventions: {self.governor_thread_obj.intervention_count}")
        
        print("\n" + "=" * 70)


# ==========================================
# 7. Example Usage
# ==========================================

def example_work_function(step: int) -> Dict[str, Any]:
    """
    Example work function that agent executes.
    
    Simulates different behaviors for different steps.
    """
    # Simulate work
    time.sleep(0.3)
    
    # Simulate different work types
    if step == 1:
        result = {"status": "quick", "data": f"Step {step} result"}
    elif step == 2:
        time.sleep(0.5)  # Step 2 takes longer
        result = {"status": "slow", "data": f"Step {step} result"}
    elif step == 5:
        result = {"status": "success", "data": f"Step {step} result"}
    else:
        result = {"status": "normal", "data": f"Step {step} result"}
    
    return result


if __name__ == "__main__":
    # Create and run supervised agent
    supervised = SupervisedAgent(
        work_function=example_work_function,
        total_steps=10,
    )
    
    supervised.run()
    
    print("\n✅ All threads complete. Supervised execution finished.")
