"""
Quick Start Guide - Integrating Sentinel with MAF

This guide shows how to add Sentinel integration to an existing MAF workflow
with minimal changes.
"""

# ============================================================================
# STEP 1: Basic Setup (No changes to existing MAF code yet)
# ============================================================================

from MAF.global_.orchestrator_graph import Global_Orchestrator
from MAF.integration import MAFSentinelRuntimeBridge
from sentinal.policies.exponential import ExponentialBackoffRetryPolicy


async def example_basic_setup():
    """
    Minimal example: Execute MAF workflow with Sentinel integration.
    
    This can be used as a drop-in replacement for existing MAF execution.
    """
    # Create MAF orchestrator (as before)
    orchestrator = Global_Orchestrator(
        auto_load_skills=True,
        db_path="subgraph_registry.db",
    )
    
    # Create Sentinel runtime bridge
    bridge = MAFSentinelRuntimeBridge(
        orchestrator=orchestrator,
        retry_policy=ExponentialBackoffRetryPolicy(max_retries=3),
        max_concurrent_skills=4,
    )
    
    # Start the runtime
    await bridge.start()
    
    # Execute workflow
    result = await bridge.execute_workflow(
        user_query="What's the weather today?",
        memory_info={"user_id": "user123"},
    )
    
    # Use result as you normally would
    print(f"Response: {result['response']}")
    print(f"Status: {result['status']}")
    print(f"Took {result['execution_time_ms']}ms")
    
    # Stop the runtime
    await bridge.stop()
    
    return result


# ============================================================================
# STEP 2: Using with Sentinel Dispatcher (Replaces asyncio.gather)
# ============================================================================

from MAF.MAFI.subgraphs.dispatcher_sentinel import dispatch_selected_skill_sentinel
from MAF.MAFI.subgraphs.subgraph_registry import SubgraphRegistry


async def example_sentinel_dispatcher():
    """
    Use the new Sentinel-aware dispatcher for skill execution.
    
    Benefits:
    - Governor-driven retries automatically applied
    - Better observability
    - State synchronized between MAF and Sentinel
    """
    # Create bridge
    orchestrator = Global_Orchestrator()
    bridge = MAFSentinelRuntimeBridge(orchestrator)
    await bridge.start()
    
    # Prepare state with skills to execute
    state = {
        "user_query": "Find me information about Python",
        "selected_skills": ["web_search", "summarize"],  # Skills to execute
    }
    
    # Use Sentinel-aware dispatcher instead of original
    result = await dispatch_selected_skill_sentinel(
        state=state,
        runtime_bridge=bridge,  # ← Main difference: pass runtime_bridge
        auto_load_from_db=True,
        db_path="subgraph_registry.db",
        state_manager=orchestrator.state_manager,
    )
    
    # Result format is compatible with original
    print(f"Executed: {result['executed_skills']}")
    print(f"Failed: {len(result['executions']) - len(result['executed_skills'])}")
    
    # NEW: Access Sentinel metrics
    metrics = result.get('sentinel_metrics', {})
    print(f"Retry policy: {metrics.get('retry_policy')}")
    print(f"Total time: {metrics.get('total_duration_ms')}ms")
    
    await bridge.stop()
    
    return result


# ============================================================================
# STEP 3: Custom Retry Policy
# ============================================================================

from sentinal.policies.retry import RetryPolicy


class MyCustomRetryPolicy(RetryPolicy):
    """Custom retry policy for specific use cases."""
    
    async def should_retry(self, execution_id, attempt, error):
        """
        Decide whether to retry based on error type and attempt count.
        """
        error_str = str(error).lower()
        
        # Always retry on timeout (up to 5 times)
        if "timeout" in error_str:
            return attempt < 5
        
        # Retry once on rate limit
        if "rate limit" in error_str:
            return attempt < 2
        
        # Don't retry other errors
        return False


async def example_custom_retry_policy():
    """Execute with custom retry policy."""
    orchestrator = Global_Orchestrator()
    
    bridge = MAFSentinelRuntimeBridge(
        orchestrator=orchestrator,
        retry_policy=MyCustomRetryPolicy(),  # ← Use custom policy
    )
    
    await bridge.start()
    
    result = await bridge.execute_workflow(
        user_query="Call an unreliable API",
        memory_info={},
    )
    
    await bridge.stop()
    return result


# ============================================================================
# STEP 4: Monitoring and Observability
# ============================================================================

async def example_monitoring():
    """Monitor execution with Sentinel integration."""
    orchestrator = Global_Orchestrator()
    
    bridge = MAFSentinelRuntimeBridge(
        orchestrator=orchestrator,
        max_concurrent_skills=4,
    )
    
    await bridge.start()
    
    # Get runtime statistics
    stats = bridge.get_execution_stats()
    print(f"Runtime stats: {stats}")
    
    # Execute and capture full result
    result = await bridge.execute_workflow(
        user_query="Test query",
        memory_info={"session_id": "session123"},
    )
    
    # Rich result information
    print(f"Execution ID: {result['execution_id']}")
    print(f"Status: {result['status']}")
    print(f"Duration: {result['execution_time_ms']}ms")
    
    if result.get('state'):
        state = result['state']
        print(f"Skills executed: {len(state.get('executions', []))}")
    
    await bridge.stop()
    return result


# ============================================================================
# STEP 5: State Inspection
# ============================================================================

from MAF.integration import StateManagerBridge


async def example_state_inspection():
    """Access and inspect synchronized state."""
    orchestrator = Global_Orchestrator()
    bridge = MAFSentinelRuntimeBridge(orchestrator)
    
    await bridge.start()
    
    # Create a state bridge for inspection
    from sentinal.core.execution.session import ExecutionSession
    from sentinal.core.execution_context import ExecutionContext
    
    session = ExecutionSession(
        execution_id="test",
        context=ExecutionContext(),
    )
    
    state_bridge = StateManagerBridge(
        state_manager=orchestrator.state_manager,
        execution_session=session,
    )
    
    # Sync state
    state_bridge.sync_to_session()
    
    # Get combined view
    combined = state_bridge.get_combined_state()
    print(f"MAF state: {combined['maf']}")
    print(f"Sentinel info: {combined['sentinel']}")
    
    await bridge.stop()
    return combined


# ============================================================================
# STEP 6: Concurrent Skill Execution
# ============================================================================

async def example_concurrent_skills():
    """Execute multiple skills concurrently with Sentinel."""
    orchestrator = Global_Orchestrator()
    bridge = MAFSentinelRuntimeBridge(orchestrator)
    
    await bridge.start()
    
    # Get skill registry
    registry = SubgraphRegistry.list_all()
    
    # Prepare skills for concurrent execution
    skills_to_run = [
        (name, entry) 
        for name, entry in list(registry.items())[:3]  # First 3 skills
    ]
    
    if skills_to_run:
        # Execute all concurrently (with retries)
        results = await bridge.execute_skills_concurrent(
            skills=skills_to_run,
            state={"context": "test"},
        )
        
        # Results include execution metadata
        for result in results:
            print(f"{result['skill_name']}: {result['status']}")
    
    await bridge.stop()
    return results


# ============================================================================
# STEP 7: Integration into Existing Orchestrator
# ============================================================================

async def example_modify_orchestrator():
    """
    Example: Modify an existing orchestrator to use Sentinel.
    
    This shows how to integrate with an existing workflow.
    """
    from MAF.global_.orchestrator_graph import Global_Orchestrator
    
    class SentinelEnabledOrchestrator(Global_Orchestrator):
        """Extend Global_Orchestrator with Sentinel integration."""
        
        def __init__(self, *args, use_sentinel=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_sentinel = use_sentinel
            self.sentinel_bridge = None
        
        async def start_sentinel(self):
            """Start Sentinel integration."""
            if self.use_sentinel:
                self.sentinel_bridge = MAFSentinelRuntimeBridge(
                    orchestrator=self,
                    retry_policy=ExponentialBackoffRetryPolicy(max_retries=3),
                )
                await self.sentinel_bridge.start()
        
        async def stop_sentinel(self):
            """Stop Sentinel integration."""
            if self.sentinel_bridge:
                await self.sentinel_bridge.stop()
        
        async def execute_with_sentinel(self, user_query, memory_info=None):
            """Execute workflow with Sentinel."""
            if not self.sentinel_bridge:
                await self.start_sentinel()
            
            return await self.sentinel_bridge.execute_workflow(
                user_query=user_query,
                memory_info=memory_info or {},
            )
    
    # Usage
    orchestrator = SentinelEnabledOrchestrator(use_sentinel=True)
    await orchestrator.start_sentinel()
    
    result = await orchestrator.execute_with_sentinel(
        user_query="Hello!",
        memory_info={},
    )
    
    await orchestrator.stop_sentinel()
    
    return result


# ============================================================================
# STEP 8: Error Handling
# ============================================================================

async def example_error_handling():
    """Proper error handling with integration."""
    orchestrator = Global_Orchestrator()
    bridge = MAFSentinelRuntimeBridge(orchestrator)
    
    try:
        await bridge.start()
        
        result = await bridge.execute_workflow(
            user_query="Could fail",
            memory_info={},
        )
        
        # Check for failures
        if result['status'] == 'failed':
            print(f"Error: {result.get('error')}")
        else:
            print(f"Success: {result['response']}")
    
    except Exception as e:
        print(f"Runtime error: {e}")
    
    finally:
        await bridge.stop()
    
    return result


# ============================================================================
# Main: Run examples
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def run_all_examples():
        print("=" * 60)
        print("MAF + Sentinel Integration Examples")
        print("=" * 60)
        
        examples = [
            ("Basic Setup", example_basic_setup),
            ("Custom Retry Policy", example_custom_retry_policy),
            ("Monitoring", example_monitoring),
            ("Error Handling", example_error_handling),
        ]
        
        for name, func in examples:
            print(f"\n--- {name} ---")
            try:
                # Note: These require actual MAF/Sentinel setup
                # Uncomment to run individual examples
                # result = await func()
                # print(f"✓ {name} completed")
                print(f"(skipped - requires full setup)")
            except Exception as e:
                print(f"✗ {name} failed: {e}")
    
    asyncio.run(run_all_examples())
