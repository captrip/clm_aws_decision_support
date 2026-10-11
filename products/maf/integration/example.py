"""
Integration Example - Using MAF with Sentinel

Demonstrates how to use the MAFSentinelRuntimeBridge to execute
MAF workflows with Sentinel's reliability, observability, and retry features.

Usage:
    python -m MAF.integration.example
    
    or in an async context:
    
    from MAF.integration.example import run_example
    await run_example()
"""

import asyncio
from typing import Dict, Any

from MAF.integration import MAFSentinelRuntimeBridge
from sentinal.policies.exponential import ExponentialRetryPolicy


async def run_example():
    """Example: Execute a workflow using Sentinel-MAF integration."""
    
    print("=" * 60)
    print("MAF + Sentinel Integration Example")
    print("=" * 60)
    
    try:
        # Import here to avoid circular imports
        from MAF.global_.orchestrator_graph import Global_Orchestrator
        
        # Create MAF orchestrator
        print("\n1. Creating MAF Global_Orchestrator...")
        orchestrator = Global_Orchestrator(
            auto_load_skills=True,
            db_path="subgraph_registry.db",
        )
        print("   ✓ Orchestrator created")
        
        # Create Sentinel runtime bridge
        print("\n2. Creating MAFSentinelRuntimeBridge...")
        retry_policy = ExponentialRetryPolicy(
            max_attempts=3,
            base_delay=0.1,
            max_delay=5,
        )
        
        bridge = MAFSentinelRuntimeBridge(
            orchestrator=orchestrator,
            state_manager=orchestrator.state_manager,
            retry_policy=retry_policy,
            max_concurrent_skills=4,
            auto_load_skills=True,
            db_path="subgraph_registry.db",
        )
        print("   ✓ Bridge created with ExponentialRetryPolicy")
        
        # Start Sentinel runtime
        print("\n3. Starting Sentinel runtime...")
        await bridge.start()
        print("   ✓ Runtime started")
        
        # Display stats
        stats = bridge.get_execution_stats()
        print(f"\n   Runtime Stats:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        # Execute a workflow
        print("\n4. Executing MAF workflow with Sentinel...")
        print("   Query: 'Hello, what can you do?'")
        
        result = await bridge.execute_workflow(
            user_query="Hello, what can you do?",
            memory_info={
                "user_id": "example_user",
                "session_id": "example_session",
            },
        )
        
        # Display results
        print("\n5. Execution Complete!")
        print(f"   Status: {result.get('status')}")
        print(f"   Execution ID: {result.get('execution_id')}")
        print(f"   Duration: {result.get('execution_time_ms', 0):.2f}ms")
        
        if result.get("status") == "completed":
            response = result.get("response", "")
            if response:
                print(f"   Response (first 200 chars): {response[:200]}...")
            else:
                print("   Response: (empty)")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Stop runtime
        print("\n6. Stopping Sentinel runtime...")
        await bridge.stop()
        print("   ✓ Runtime stopped")
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
        return result
    
    except ImportError as e:
        print(f"\nError: Could not import required modules")
        print(f"  {e}")
        print("\nPlease ensure MAF and Sentinel modules are in PYTHONPATH")
    
    except Exception as e:
        print(f"\nError during execution:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


async def run_example_with_skills():
    """
    Example: Execute individual skills via Sentinel.
    
    Demonstrates the lower-level skill execution API.
    """
    print("\n" + "=" * 60)
    print("Skill Execution Example")
    print("=" * 60)
    
    try:
        from MAF.global_.orchestrator_graph import Global_Orchestrator
        from MAF.MAFI.subgraphs.subgraph_registry import SubgraphRegistry
        
        # Create orchestrator and bridge
        orchestrator = Global_Orchestrator()
        bridge = MAFSentinelRuntimeBridge(orchestrator)
        
        # Start runtime
        await bridge.start()
        
        # Get available skills
        print("\n1. Available skills:")
        registry = SubgraphRegistry.list_all()
        
        if not registry:
            print("   (No skills registered)")
        else:
            for skill_name in list(registry.keys())[:5]:  # Show first 5
                print(f"   - {skill_name}")
            if len(registry) > 5:
                print(f"   ... and {len(registry) - 5} more")
        
        # Execute a skill if any available
        if registry:
            skill_name = list(registry.keys())[0]
            print(f"\n2. Executing skill: {skill_name}")
            
            skill_entry = registry[skill_name]
            test_state = {
                "user_query": "Test query",
                "context": "Test context",
            }
            
            result = await bridge.execute_skill(
                skill_name=skill_name,
                skill_entry=skill_entry,
                state=test_state,
            )
            
            print(f"   Status: {result.get('status')}")
            print(f"   Duration: {result.get('execution_time_ms', 0):.2f}ms")
            if result.get("error"):
                print(f"   Error: {result.get('error')}")
        
        # Stop runtime
        await bridge.stop()
        print("\n✓ Skill execution example completed")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_example())
