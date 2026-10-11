"""
Sentinel-Integrated MAF Orchestrator

Production orchestrator with 3-thread supervision (Actor|Observer|Governor)
built into every agent execution.
"""

from typing import Any, Dict, Optional
import asyncio

# Handle imports for both notebook and script contexts
try:
    from MAF.sentinel_maf_integration import SentinelMAFAgent
except ModuleNotFoundError:
    from sentinel_maf_integration import SentinelMAFAgent


class SentinelOrchestrator:
    """
    Production MAF Orchestrator with Sentinel 3-thread supervision.

    Every agent automatically runs with:
    - Actor thread: Executes the agent
    - Observer thread: Watches runtime_memory
    - Governor thread: Can intervene with instructions

    Usage:
        from MAF.global_.sentinel_orchestrator import SentinelOrchestrator

        orc = SentinelOrchestrator(
            auto_load_skills=True,
            db_path="subgraph_registry.db",
        )

        result = await orc.ainvoke({
            "user_id": "user_123",
            "user_query": "What's the weather?",
        })
    """

    def __init__(
        self,
        auto_load_skills: bool = True,
        db_path: str = "subgraph_registry.db",
        SubgraphFactory: Any = None,
        quality_threshold: float = 0.75,
        enable_supervision: bool = True,
    ):
        """
        Initialize Sentinel-integrated orchestrator.

        Args:
            auto_load_skills: Auto-load skills from registry
            db_path: Path to skill registry database
            SubgraphFactory: Factory for creating subgraphs
            quality_threshold: Quality threshold for Governor interventions
            enable_supervision: Enable/disable 3-thread supervision
        """
        self.auto_load_skills = auto_load_skills
        self.db_path = db_path
        self.SubgraphFactory = SubgraphFactory
        self.quality_threshold = quality_threshold
        self.enable_supervision = enable_supervision

        # Will hold supervised agents
        self.agents = {}

        # Metrics
        self.execution_count = 0
        self.total_observations = 0
        self.total_interventions = 0

    def build(self):
        """
        Build the orchestrator with supervised agents.

        Returns:
            self (for chaining)
        """

        # Import MAF components
        try:
            # Try notebook-style import first
            try:
                from MAFI.agents.IU_agents import IntentUnderstandingAgent
            except ModuleNotFoundError:
                from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent

            try:
                from global_.state import Orchestrator_Global_State
            except ModuleNotFoundError:
                from MAF.global_.state import Orchestrator_Global_State

            # Create agents with supervision
            if self.enable_supervision:
                self.agents['intent'] = SentinelMAFAgent(
                    maf_agent=IntentUnderstandingAgent(),
                    agent_name="Intent Understanding",
                    quality_threshold=self.quality_threshold,
                    timeout_sec=60.0,
                )
            else:
                self.agents['intent'] = IntentUnderstandingAgent()

            print(f"✓ Built orchestrator with {'supervised' if self.enable_supervision else 'standard'} agents")

        except ImportError as e:
            print(f"⚠️  Could not load MAF agents: {e}")
            print("   Orchestrator built without agents (add agents manually)")

        return self

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the orchestration pipeline with supervision.

        Args:
            state: Initial state with user_query, user_id, etc.

        Returns:
            Final orchestrator state with results and supervision metadata
        """

        self.execution_count += 1

        # Use any manually added agents first
        agent_key = list(self.agents.keys())[0] if self.agents else None

        # Stage 1: Intent Understanding (with supervision)
        if agent_key:
            if self.enable_supervision:
                # Get full supervision metadata
                intent_result = await self.agents[agent_key].execute(state)

                # Track metrics
                self.total_observations += len(intent_result['supervision']['observer_observations'])
                self.total_interventions += len(intent_result['supervision']['governor_interventions'])

                # Extract the actual MAF result
                intent_understanding = intent_result['result']
                if intent_understanding is None:
                    intent_understanding = {"intent_understanding": "Agent returned no result"}

                supervision_metadata = intent_result['supervision']
            else:
                # Standard execution (no supervision)
                intent_understanding = await self.agents[agent_key](state)
                if intent_understanding is None:
                    intent_understanding = {"intent_understanding": "Agent returned no result"}

                supervision_metadata = None
        else:
            intent_understanding = {"intent_understanding": "No intent agent configured"}
            supervision_metadata = None

        # Add more stages here (action planning, execution, etc.)

        # Final result
        final_answer_text = ""
        if intent_understanding:
            final_answer_text = intent_understanding.get("intent_understanding", "")

        return {
            "user_query": state.get("user_query"),
            "intent_understanding": intent_understanding,
            "final_answer": [final_answer_text],
            "supervision_enabled": self.enable_supervision,
            "supervision_metadata": supervision_metadata if self.enable_supervision else None,
            "orchestrator_stats": {
                "total_executions": self.execution_count,
                "total_observations": self.total_observations,
                "total_interventions": self.total_interventions,
            }
        }

    def add_supervised_agent(self, agent_name: str, maf_agent: Any, quality_threshold: Optional[float] = None):
        """
        Add a supervised agent to the orchestrator.

        Args:
            agent_name: Name for the agent (e.g., "Intent Understanding")
            maf_agent: The MAF agent to wrap
            quality_threshold: Optional quality threshold (defaults to orchestrator's)
        """

        if quality_threshold is None:
            quality_threshold = self.quality_threshold

        if self.enable_supervision:
            self.agents[agent_name.lower().replace(" ", "_")] = SentinelMAFAgent(
                maf_agent=maf_agent,
                agent_name=agent_name,
                quality_threshold=quality_threshold,
            )
        else:
            self.agents[agent_name.lower().replace(" ", "_")] = maf_agent

        print(f"✓ Added {'supervised' if self.enable_supervision else 'standard'} agent: {agent_name}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""

        return {
            "execution_count": self.execution_count,
            "total_observations": self.total_observations,
            "total_interventions": self.total_interventions,
            "supervision_enabled": self.enable_supervision,
            "agents": list(self.agents.keys()),
        }
