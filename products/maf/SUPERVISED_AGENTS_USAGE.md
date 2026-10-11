"""
HOW TO USE: Supervised Agents in MAF Orchestrator

Quick integration guide for wrapping MAF agents with supervision.
"""

# ==========================================
# OPTION 1: Direct Replacement
# ==========================================
"""
In your orchestrator initialization, replace the agent with supervised version.

FILE: global_/orchestrator_graph.py (around line 27)

BEFORE:
    class Global_Orchestrator:
        def __init__(self, ...):
            ...
            self.intent_understanding = IntentUnderstandingAgent()
            self.action_extractor = ActionExtractorAgent()
            ...

AFTER:
    from MAF.supervised_agents_pilot import SupervisedIntentUnderstandingAgent
    
    class Global_Orchestrator:
        def __init__(self, ...):
            ...
            # Wrap with supervision
            original_intent = IntentUnderstandingAgent()
            self.intent_understanding = SupervisedIntentUnderstandingAgent(original_intent)
            
            self.action_extractor = ActionExtractorAgent()
            ...

That's it! The rest of the code stays the same.
The orchestrator will automatically use supervision.
"""


# ==========================================
# OPTION 2: Feature Flag
# ==========================================
"""
Add supervision as an optional feature that can be toggled.

FILE: global_/orchestrator_graph.py (around line 18)

    class Global_Orchestrator:
        def __init__(
            self,
            auto_load_skills: bool = True,
            db_path: str = "subgraph_registry.db",
            SubgraphFactory = None,
            tools_registry = None,
            use_supervision: bool = False,  # <-- NEW PARAMETER
        ):
            self.use_supervision = use_supervision
            
            # Create agents
            self.intent_understanding = IntentUnderstandingAgent()
            self.action_extractor = ActionExtractorAgent()
            
            # Optionally wrap with supervision
            if self.use_supervision:
                from MAF.supervised_agents_pilot import SupervisedIntentUnderstandingAgent
                self.intent_understanding = SupervisedIntentUnderstandingAgent(
                    self.intent_understanding
                )

Usage:
    # Without supervision (existing behavior)
    orchestrator = Global_Orchestrator()
    
    # With supervision
    orchestrator = Global_Orchestrator(use_supervision=True)
"""


# ==========================================
# OPTION 3: Per-Agent Control
# ==========================================
"""
Fine-grained control - enable supervision for specific agents.

    from MAF.supervised_agents_pilot import SupervisedIntentUnderstandingAgent
    from MAF.supervised_subagents import wrap_maf_subagent
    
    orchestrator = Global_Orchestrator()
    
    # Wrap only intent understanding with supervision
    orchestrator.intent_understanding = SupervisedIntentUnderstandingAgent(
        orchestrator.intent_understanding
    )
    
    # Action extractor remains unsupervised
    # Use as normal
    result = await orchestrator.action_extractor(state)
"""


# ==========================================
# OPTION 4: Custom Governance Rules
# ==========================================
"""
Extend SupervisedIntentUnderstandingAgent for custom rules.

    from MAF.supervised_agents_pilot import SupervisedIntentUnderstandingAgent
    from experiment.agent_supervision import CorrectionType
    from MAF.supervised_subagents import SubagentGovernor
    
    class CustomGovernor(SubagentGovernor):
        \"\"\"Custom rules for your specific use case.\"\"\"
        
        def _apply_subagent_rules(self, snapshot):
            metrics = snapshot.get("metrics", {})
            quality = metrics.get("quality_score", 1.0)
            
            # Your custom rules here
            if quality < 0.4:
                self._request_intervention(
                    CorrectionType.ESCALATE,
                    "Quality critically low - escalating to human",
                    priority=5
                )
    
    # Use custom governor
    class CustomSupervisedIntent(SupervisedIntentUnderstandingAgent):
        def __call__(self, state):
            # ... copy from SupervisedIntentUnderstandingAgent
            # ... but use CustomGovernor instead
            self.governor = CustomGovernor(self.state, "intent")
            # ... rest of implementation
"""


# ==========================================
# COMPLETE EXAMPLE
# ==========================================
"""
Complete working example showing all pieces together.

FILE: main_orchestrator.py (or modify global_/orchestrator_graph.py)
"""

import asyncio
from typing import Optional, Dict, Any


# Step 1: Import needed classes
from MAF.supervised_agents_pilot import SupervisedIntentUnderstandingAgent


async def example_full_orchestrator():
    \"\"\"
    Complete example: Run MAF orchestrator with supervised intent agent.
    \"\"\"
    
    # Step 2: Import the real orchestrator
    from global_.orchestrator_graph import Global_Orchestrator
    
    # Step 3: Create orchestrator (no changes needed to orchestrator code)
    orchestrator = Global_Orchestrator()
    
    # Step 4: Wrap intent agent with supervision
    orchestrator.intent_understanding = SupervisedIntentUnderstandingAgent(
        orchestrator.intent_understanding
    )
    
    print(f"✓ Orchestrator created with supervision enabled")
    print(f"  Intent Agent: {orchestrator.intent_understanding.name}")
    print(f"  Supervision: {'ENABLED' if isinstance(orchestrator.intent_understanding, SupervisedIntentUnderstandingAgent) else 'DISABLED'}")
    
    # Step 5: Use orchestrator normally
    state = {
        "user_query": "What's the weather?",
        "updated_query": "What's the weather?",
        "user_id": "user_123",
        "session_id": "session_456",
    }
    
    print(f"\n🚀 Executing orchestrator...")
    
    try:
        # Call orchestrator (will use supervised intent agent automatically)
        # Note: This would call the full orchestrator flow
        # For demo, we just show the intent stage
        
        intent_result = await orchestrator.intent_understanding(state)
        print(f"✓ Intent result: {intent_result}")
        
    except Exception as e:
        print(f"Error: {e}")


# ==========================================
# MONITORING & METRICS
# ==========================================
"""
Access supervision metrics from your agents.

    supervised_agent = SupervisedIntentUnderstandingAgent(original_agent)
    
    # After execution
    print(f"Executions: {supervised_agent.execution_count}")
    print(f"Interventions: {supervised_agent.intervention_count}")
    
    # Access detailed state
    snapshot = supervised_agent.state.get_snapshot()
    print(f"Status: {snapshot['status']}")
    print(f"Metrics: {snapshot['metrics']}")
    print(f"Decisions: {snapshot['decisions']}")
"""


# ==========================================
# TESTING
# ==========================================
"""
Test your supervised agents.

    # In your test file
    import pytest
    
    @pytest.mark.asyncio
    async def test_supervised_intent_agent():
        from MAF.supervised_agents_pilot import SupervisedIntentUnderstandingAgent
        
        agent = SupervisedIntentUnderstandingAgent()
        
        result = await agent({
            "user_query": "Test query",
            "updated_query": "Test query",
        })
        
        assert "intent_understanding" in result
        assert agent.execution_count == 1
"""


# ==========================================
# DEPLOYMENT
# ==========================================
"""
Production deployment considerations.

1. FEATURE FLAGS
   - Use environment variables to enable/disable supervision
   - Example: ENABLE_SUPERVISION=true
   
2. GRADUAL ROLLOUT
   - Start with monitoring-only mode (no interventions)
   - Gradually enable automated interventions
   - Monitor metrics and adjust rules
   
3. LOGGING
   - Capture all supervision events
   - Track intervention types and frequencies
   - Use for tuning governance rules
   
4. PERFORMANCE
   - Supervision adds threads (minimal overhead)
   - Observer thread: ~1% CPU
   - Governor thread: ~1-2% CPU
   - Total: <5% overhead for typical use
   
5. CONFIGURATION
   - Make thresholds configurable
   - Example: quality_threshold, latency_threshold
   - Allow runtime rule updates
"""


if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "=" * 70)
    print("INTEGRATION OPTIONS SUMMARY")
    print("=" * 70)
    
    print("""
    OPTION 1: Direct Replacement (Simplest)
    - Wrap agent once during initialization
    - No other code changes needed
    - Recommended for quick integration
    
    OPTION 2: Feature Flag (Flexible)
    - Pass use_supervision parameter
    - Easy to toggle on/off
    - Good for A/B testing
    
    OPTION 3: Per-Agent Control (Fine-Grained)
    - Wrap only specific agents
    - Mix supervised and unsupervised
    - Maximum flexibility
    
    OPTION 4: Custom Governance (Advanced)
    - Extend base classes
    - Implement custom rules
    - Domain-specific supervision
    
    RECOMMENDATION FOR YOUR USE CASE:
    1. Start with OPTION 1 (Direct Replacement)
    2. Test with test_supervised_agents.py
    3. Move to OPTION 2 (Feature Flag) for production
    4. Gradually add custom rules (OPTION 4) as needed
    """)
    
    print("\nNEXT STEPS:")
    print("1. Choose integration option above")
    print("2. Modify global_/orchestrator_graph.py")
    print("3. Run MAF workflow tests")
    print("4. Monitor metrics and adjust rules")
