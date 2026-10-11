"""
Sentinel + MAF Integration Framework

Integrates the 3-thread architecture (Actor, Observer, Governor)
directly into MAF as a unified framework.

Every MAF agent automatically runs with 3 threads:
1. Actor Thread - Executes the agent
2. Observer Thread - Watches runtime_memory
3. Governor Thread - Can intervene with instructions
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import threading
import time
import asyncio
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, UTC
from abc import ABC, abstractmethod


# ==========================================
# Core: Execution State (Shared Memory)
# ==========================================

@dataclass
class SentinelExecutionState:
    """
    Shared execution state between all 3 threads.

    This is the memory that:
    - Actor updates during execution
    - Observer watches for changes
    - Governor reads to make decisions
    """

    # Execution metadata
    execution_id: str
    agent_name: str
    status: str = "PENDING"
    attempt: int = 1

    # Runtime memory (what Observer watches)
    runtime_memory: Dict[str, Any] = field(default_factory=dict)

    # Instructions from Governor to Actor
    runtime_instructions: list[str] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: Optional[datetime] = None

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_memory(self, key: str, value: Any):
        """Thread-safe memory update (Actor calls this)."""
        with self._lock:
            self.runtime_memory[key] = value

    def get_memory(self, key: str, default=None):
        """Thread-safe memory read (Observer/Governor calls this)."""
        with self._lock:
            return self.runtime_memory.get(key, default)

    def add_instruction(self, instruction: str):
        """Thread-safe instruction add (Governor calls this)."""
        with self._lock:
            self.runtime_instructions.append(instruction)

    def get_instructions(self):
        """Thread-safe instruction read (Actor calls this)."""
        with self._lock:
            return self.runtime_instructions.copy()

    def transition(self, status: str):
        """Thread-safe status transition."""
        with self._lock:
            self.status = status
            if status in ["COMPLETED", "FAILED"]:
                self.completed_at = datetime.now(UTC)


# ==========================================
# Thread 1: Actor (Executes Agent)
# ==========================================

class ActorThread(threading.Thread):
    """
    Actor thread that executes the MAF agent.

    This is the "normal flow" - the agent executing as usual,
    but with memory updates that Observer watches.
    """

    def __init__(
        self,
        maf_agent: Any,
        state: SentinelExecutionState,
        maf_input: Dict[str, Any],
    ):
        super().__init__(name=f"Actor-{state.execution_id}", daemon=False)
        self.maf_agent = maf_agent
        self.state = state
        self.maf_input = maf_input
        self.result = None
        self.error = None

    def run(self):
        """Execute the MAF agent (runs in separate OS thread)."""

        # Update memory: starting
        self.state.update_memory("actor_status", "starting")
        self.state.update_memory("thread_id", threading.get_ident())
        self.state.transition("RUNNING")

        # Check for instructions from Governor
        instructions = self.state.get_instructions()
        if instructions:
            self.state.update_memory("following_instructions", instructions)

        # Execute
        start_time = time.time()

        try:
            self.state.update_memory("actor_status", "executing")

            # Run the MAF agent
            result = asyncio.run(self.maf_agent(self.maf_input))

            latency_ms = (time.time() - start_time) * 1000

            # Update memory with results (Observer sees this!)
            self.state.update_memory("actor_status", "complete")
            self.state.update_memory("result", result)
            self.state.update_memory("latency_ms", latency_ms)
            self.state.transition("COMPLETED")

            self.result = result

        except Exception as e:
            self.state.update_memory("actor_status", "error")
            self.state.update_memory("error", str(e))
            self.state.transition("FAILED")
            self.error = e


# ==========================================
# Thread 2: Observer (Watches Memory)
# ==========================================

class ObserverThread(threading.Thread):
    """
    Observer thread that watches runtime_memory changes.

    Logs changes, collects metrics, can trigger alerts.
    """

    def __init__(self, state: SentinelExecutionState, duration_sec: float = 10.0):
        super().__init__(name=f"Observer-{state.execution_id}", daemon=False)
        self.state = state
        self.duration_sec = duration_sec
        self.observations = []
        self.running = True

    def run(self):
        """Watch memory (runs in separate OS thread)."""

        start_time = time.time()
        last_status = None

        while self.running and (time.time() - start_time) < self.duration_sec:

            # Read current state
            current_status = self.state.get_memory("actor_status")

            if current_status != last_status:
                # State changed!
                observation = {
                    "timestamp": time.time(),
                    "status": current_status,
                    "memory_snapshot": self.state.get_memory("result"),
                    "latency": self.state.get_memory("latency_ms"),
                }

                self.observations.append(observation)
                last_status = current_status

            # Check if done
            if self.state.status in ["COMPLETED", "FAILED"]:
                break

            time.sleep(0.05)  # Poll every 50ms

        self.running = False

    def stop(self):
        self.running = False


# ==========================================
# Thread 3: Governor (Intervenes)
# ==========================================

class GovernorThread(threading.Thread):
    """
    Governor thread that monitors and can intervene.

    Can issue:
    - RETRY_WITH_FALLBACK
    - IMPROVE_QUALITY
    - USE_BETTER_TOOLS
    - CHANGE_PARAMETERS
    """

    def __init__(
        self,
        state: SentinelExecutionState,
        duration_sec: float = 10.0,
        quality_threshold: float = 0.7,
    ):
        super().__init__(name=f"Governor-{state.execution_id}", daemon=False)
        self.state = state
        self.duration_sec = duration_sec
        self.quality_threshold = quality_threshold
        self.interventions = []
        self.running = True

    def run(self):
        """Monitor and intervene (runs in separate OS thread)."""

        start_time = time.time()

        while self.running and (time.time() - start_time) < self.duration_sec:

            actor_status = self.state.get_memory("actor_status")

            # Intervention logic
            if actor_status == "error":
                # Actor failed - suggest retry
                self.state.add_instruction("RETRY_WITH_FALLBACK")

                self.interventions.append({
                    "type": "retry",
                    "reason": "actor_error",
                })

            elif actor_status == "complete":
                # Check quality
                result = self.state.get_memory("result", {})
                quality = self._compute_quality(result)

                if quality < self.quality_threshold:
                    self.state.add_instruction("IMPROVE_QUALITY")
                    self.state.add_instruction("USE_BETTER_TOOLS")

                    self.interventions.append({
                        "type": "quality_improvement",
                        "quality": quality,
                        "threshold": self.quality_threshold,
                    })

            # Check if done
            if self.state.status in ["COMPLETED", "FAILED"]:
                break

            time.sleep(0.1)  # Poll every 100ms

        self.running = False

    def _compute_quality(self, result: Dict[str, Any]) -> float:
        """Compute quality from result."""
        # Customize based on agent type
        if "intent_understanding" in result:
            text = result.get("intent_understanding", "")
            if not text:
                return 0.2
            if len(str(text).strip()) > 20:
                return 0.9
            return 0.5

        # Default: assume good quality if result exists
        return 0.8 if result else 0.2

    def stop(self):
        self.running = False


# ==========================================
# Unified Framework: SentinelMAFAgent
# ==========================================

class SentinelMAFAgent:
    """
    Wrapper that runs ANY MAF agent with 3-thread supervision.

    Usage:
        from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent

        # Create regular MAF agent
        maf_agent = IntentUnderstandingAgent()

        # Wrap with Sentinel
        sentinel_agent = SentinelMAFAgent(
            maf_agent=maf_agent,
            agent_name="Intent Understanding",
            quality_threshold=0.7,
        )

        # Use normally - 3 threads run automatically
        result = await sentinel_agent.execute(maf_input)
    """

    def __init__(
        self,
        maf_agent: Any,
        agent_name: str,
        quality_threshold: float = 0.7,
        timeout_sec: float = 30.0,
        enable_observer: bool = True,
        enable_governor: bool = True,
    ):
        self.maf_agent = maf_agent
        self.agent_name = agent_name
        self.quality_threshold = quality_threshold
        self.timeout_sec = timeout_sec
        self.enable_observer = enable_observer
        self.enable_governor = enable_governor

        self.execution_count = 0

    async def execute(self, maf_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the MAF agent with 3-thread supervision.

        Returns the agent result plus supervision metadata.
        """

        self.execution_count += 1
        execution_id = f"exec-{self.execution_count}-{int(time.time() * 1000)}"

        # Create shared state
        state = SentinelExecutionState(
            execution_id=execution_id,
            agent_name=self.agent_name,
        )

        # Create 3 threads
        actor = ActorThread(self.maf_agent, state, maf_input)
        observer = ObserverThread(state, self.timeout_sec) if self.enable_observer else None
        governor = GovernorThread(state, self.timeout_sec, self.quality_threshold) if self.enable_governor else None

        # Start all threads
        threads = [actor]
        if observer:
            threads.append(observer)
        if governor:
            threads.append(governor)

        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=self.timeout_sec)

        # Collect results
        result = {
            "execution_id": execution_id,
            "agent_name": self.agent_name,
            "status": state.status,
            "result": actor.result,
            "error": actor.error,
            "runtime_memory": dict(state.runtime_memory),
            "runtime_instructions": state.get_instructions(),
            "supervision": {
                "observer_observations": observer.observations if observer else [],
                "governor_interventions": governor.interventions if governor else [],
            },
            "timing": {
                "started_at": state.started_at.isoformat(),
                "completed_at": state.completed_at.isoformat() if state.completed_at else None,
                "latency_ms": state.get_memory("latency_ms"),
            },
        }

        return result

    async def __call__(self, maf_input: Dict[str, Any]) -> Dict[str, Any]:
        """Allow calling as a function (MAF-compatible interface)."""
        result = await self.execute(maf_input)
        # Return just the agent result for MAF compatibility
        return result["result"] if result["result"] else {}


# ==========================================
# Helper: Wrap Existing MAF Agent
# ==========================================

def wrap_maf_agent_with_sentinel(
    maf_agent: Any,
    agent_name: str,
    quality_threshold: float = 0.7,
    timeout_sec: float = 30.0,
) -> SentinelMAFAgent:
    """
    Convenience function to wrap any MAF agent with Sentinel.

    Example:
        from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent
        from MAF.sentinel_maf_integration import wrap_maf_agent_with_sentinel

        # Wrap the agent
        iu_agent = wrap_maf_agent_with_sentinel(
            maf_agent=IntentUnderstandingAgent(),
            agent_name="Intent Understanding",
            quality_threshold=0.75,
        )

        # Use normally
        result = await iu_agent({"user_query": "hello", "updated_query": "hello"})
    """

    return SentinelMAFAgent(
        maf_agent=maf_agent,
        agent_name=agent_name,
        quality_threshold=quality_threshold,
        timeout_sec=timeout_sec,
    )


# ==========================================
# Integration with Global Orchestrator
# ==========================================

class SentinelOrchestrator:
    """
    Drop-in replacement for Global_Orchestrator that uses Sentinel.

    All agents run with 3-thread supervision automatically.
    """

    def __init__(
        self,
        intent_agent: Any,
        planner_agent: Any,
        execution_agent: Any,
        response_agent: Any,
        quality_threshold: float = 0.7,
    ):
        # Wrap all agents with Sentinel
        self.intent_agent = wrap_maf_agent_with_sentinel(
            intent_agent, "Intent Understanding", quality_threshold
        )
        self.planner_agent = wrap_maf_agent_with_sentinel(
            planner_agent, "Action Planning", quality_threshold
        )
        self.execution_agent = wrap_maf_agent_with_sentinel(
            execution_agent, "Skill Execution", quality_threshold
        )
        self.response_agent = wrap_maf_agent_with_sentinel(
            response_agent, "Response Generation", quality_threshold
        )

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the orchestration pipeline with 3-thread supervision
        on each agent.
        """

        # Each agent runs with Actor|Observer|Governor threads
        intent_result = await self.intent_agent.execute(state)
        planner_result = await self.planner_agent.execute({**state, **intent_result["result"]})
        execution_result = await self.execution_agent.execute({**state, **planner_result["result"]})
        response_result = await self.response_agent.execute({**state, **execution_result["result"]})

        return {
            "final_answer": response_result["result"],
            "supervision_metadata": {
                "intent": intent_result["supervision"],
                "planner": planner_result["supervision"],
                "execution": execution_result["supervision"],
                "response": response_result["supervision"],
            },
        }
