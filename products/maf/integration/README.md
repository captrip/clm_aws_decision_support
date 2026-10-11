"""
MAF + Sentinel Integration Documentation

## Overview

The MAF + Sentinel integration provides a unified orchestration layer that combines:
- **MAF**: LLM-driven multi-agent workflow orchestration
- **Sentinel**: Actor-based reliable execution with retry policies

Benefits:
- 🔄 **Governor-driven retries**: Automatic retry policies for skill failures
- 📊 **Better observability**: Complete execution traces and event logs
- 🎯 **State management**: Scoped state synchronized between systems
- ⚡ **Concurrent execution**: Safe parallel skill execution via actors
- 🛡️ **Fault tolerance**: Built-in error handling and recovery

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│            MAF Global_Orchestrator                      │
│  (User query → Intent → Planning → Skills → Response)   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│        MAFSentinelRuntimeBridge (Entry Point)           │
│                                                         │
│  • Manages Sentinel runtime lifecycle                   │
│  • Wraps MAF workflows as Executables                   │
│  • Coordinates skill execution                          │
│  • Syncs state between systems                          │
└─────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┬──────────────────┐
        ↓                  ↓                  ↓
   ┌─────────┐      ┌─────────┐      ┌──────────────┐
   │ Workflow │      │ Skill   │      │ State        │
   │Executable│      │Executor │      │Bridge        │
   └─────────┘      └─────────┘      └──────────────┘
        ↓                  ↓                  ↓
   ┌─────────────────────────────────────────────────┐
   │         Sentinel Runtime                        │
   │                                                 │
   │  • Actor mailbox system                         │
   │  • Message dispatcher                           │
   │  • Governor (retry policies)                    │
   │  • Event/signal pub-sub                         │
   └─────────────────────────────────────────────────┘
```

### Key Components

#### 1. MAFWorkflowExecutable
Wraps `Global_Orchestrator.ainvoke()` as a Sentinel Executable.
- Converts MAF state to ExecutionContext
- Executes complete workflow
- Tracks total execution time
- Handles workflow-level errors

#### 2. SkillExecutable
Wraps individual skills (runner functions or LangGraph graphs).
- Manages skill input/output state
- Enriches state with StateManager context
- Tracks skill-level execution time
- Stores results in scoped state

#### 3. SkillExecutor (Actor)
Sentinel actor that executes skills with observability.
- Receives `ExecuteSkillSignal` messages
- Invokes SkillExecutable
- Publishes ExecutionCompleted/ExecutionFailed events
- Enables Governor-driven retries

#### 4. StateManagerBridge
Synchronizes MAF's StateManager with Sentinel's ExecutionSession.
- Syncs MAF state → ExecutionSession (read)
- Syncs ExecutionSession → MAF state (write)
- Stores skill results in both systems
- Provides combined state view

#### 5. MAFSentinelRuntimeBridge
Unified runtime entry point.
- Manages Sentinel runtime lifecycle (start/stop)
- Registers SkillExecutor actors
- Submits workflows for execution
- Coordinates concurrent skill execution
- Tracks execution sessions

## Usage

### Basic Workflow Execution

```python
from MAF.global_.orchestrator_graph import Global_Orchestrator
from MAF.integration import MAFSentinelRuntimeBridge
from sentinal.policies.exponential import ExponentialBackoffRetryPolicy

# Create orchestrator
orchestrator = Global_Orchestrator()

# Create runtime bridge
bridge = MAFSentinelRuntimeBridge(
    orchestrator=orchestrator,
    retry_policy=ExponentialBackoffRetryPolicy(max_retries=3),
    max_concurrent_skills=4,
)

# Start runtime
await bridge.start()

# Execute workflow
result = await bridge.execute_workflow(
    user_query="What can you help with?",
    memory_info={"user_id": "123"},
)

print(f"Status: {result['status']}")
print(f"Response: {result['response']}")
print(f"Time: {result['execution_time_ms']}ms")

# Stop runtime
await bridge.stop()
```

### Individual Skill Execution

```python
from MAF.MAFI.subgraphs.subgraph_registry import SubgraphRegistry

# Get skill from registry
registry = SubgraphRegistry.list_all()
skill_entry = registry["web_search"]

# Execute single skill
result = await bridge.execute_skill(
    skill_name="web_search",
    skill_entry=skill_entry,
    state={"query": "python async"},
)

print(f"Result: {result}")
```

### Concurrent Skill Execution

```python
# Define multiple skills to execute
skills = [
    ("web_search", registry["web_search"]),
    ("calculator", registry["calculator"]),
]

# Execute all concurrently (with retry policies)
results = await bridge.execute_skills_concurrent(
    skills=skills,
    state={"context": "..."},
)

for result in results:
    print(f"{result['skill_name']}: {result['status']}")
```

## State Management

### State Flow

```
MAF StateManager              Sentinel ExecutionSession
      ↓                               ↓
   global_state ←────────────→ runtime_data
   scope: "agent" ←─StateManager─→ runtime_data["maf_state"]
   scope: "skill1" ←────Bridge────→ runtime_data["skill_results"]
   scope: "skill2"
```

### Accessing State

```python
from MAF.integration import StateManagerBridge

bridge = StateManagerBridge(state_manager, execution_session)

# Sync both directions
bridge.sync_to_session()    # MAF → Sentinel
bridge.sync_from_session()  # Sentinel → MAF

# Store skill result
bridge.store_skill_result(
    skill_name="web_search",
    result={...},
    error=None,
)

# Get combined view
combined = bridge.get_combined_state()
print(combined["maf"]["global"])
print(combined["sentinel"]["execution_id"])
```

## Retry Policies

### Available Policies

The integration supports Sentinel's pluggable retry policies:

```python
from sentinal.policies.exponential import ExponentialBackoffRetryPolicy
from sentinal.policies.no_retry import NoRetryPolicy

# Exponential backoff (recommended for production)
policy = ExponentialBackoffRetryPolicy(
    max_retries=3,
    initial_delay_ms=100,
    max_delay_ms=5000,
    backoff_factor=2.0,
)

# No retry (fail-fast)
policy = NoRetryPolicy()

# Custom policy
class CustomRetryPolicy(RetryPolicy):
    async def should_retry(self, execution_id, attempt, error):
        # Your logic here
        return attempt < 3
```

### How Governor Uses Policies

```
Skill Execution
      ↓
   ExecutionFailed event published
      ↓
   Governor observes ExecutionFailed
      ↓
   Governor applies retry_policy.should_retry()
      ↓
   If true → publish RetryExecution signal
   If false → execution failed permanently
```

## Observability

### Execution Tracing

Every workflow execution is tracked with:
- **execution_id**: Unique identifier
- **session_id**: Session context
- **status**: 'completed', 'failed', 'retrying'
- **execution_time_ms**: Total duration
- **state snapshot**: Complete state at completion
- **events**: Full event log from execution

### Events Published

The system publishes these events via Sentinel's MessageBus:
- `ExecutionStarted`: Workflow execution begins
- `ExecutionCompleted`: Workflow completed successfully
- `ExecutionFailed`: Workflow/skill failed
- `RetryExecution`: Governor initiated retry

### Accessing Metrics

```python
# Get runtime statistics
stats = bridge.get_execution_stats()
print(f"Active sessions: {stats['active_sessions']}")
print(f"Skill executors: {stats['registered_skill_executors']}")
print(f"Retry policy: {stats['retry_policy']}")

# Get execution result
result = await bridge.execute_workflow(...)
print(f"Execution time: {result['execution_time_ms']}ms")
print(f"Execution ID: {result['execution_id']}")
```

## Migration Guide

### From asyncio.gather() to Sentinel

**Before:**
```python
from MAF.MAFI.subgraphs.dispatcher_concurrent import dispatch_selected_skill

result = await dispatch_selected_skill(state, auto_load_from_db=True, ...)
```

**After:**
```python
from MAF.MAFI.subgraphs.dispatcher_sentinel import dispatch_selected_skill_sentinel
from MAF.integration import MAFSentinelRuntimeBridge

bridge = MAFSentinelRuntimeBridge(orchestrator)
await bridge.start()

result = await dispatch_selected_skill_sentinel(
    state,
    runtime_bridge=bridge,
    auto_load_from_db=True,
    ...
)

await bridge.stop()
```

Both APIs are compatible - same return format, but with added `sentinel_metrics`:
```python
{
    "executions": [...],
    "executed_skills": [...],
    "tool_results_summary": [...],
    "sentinel_metrics": {
        "successful": 2,
        "failed": 0,
        "total_duration_ms": 1234.5,
        "retry_policy": "ExponentialBackoffRetryPolicy",
    }
}
```

## Advanced Usage

### Custom Retry Policy

```python
from sentinal.policies.retry import RetryPolicy

class CustomRetryPolicy(RetryPolicy):
    async def should_retry(self, execution_id, attempt, error):
        # Retry only on specific error types
        if "timeout" in str(error):
            return attempt < 5
        return attempt < 2

bridge = MAFSentinelRuntimeBridge(
    orchestrator=orchestrator,
    retry_policy=CustomRetryPolicy(),
)
```

### Hooking into Events

```python
from sentinal.bus.message_bus import MessageBus

# Get the message bus from runtime
bus = bridge.sentinel_runtime.dispatcher.message_bus

# Subscribe to events
async def on_skill_completed(event):
    print(f"Skill completed: {event.data}")

bus.subscribe("ExecutionCompleted", on_skill_completed)
```

### Distributed Skill Execution

Skills can be executed on different machines by:
1. Registering SkillExecutor actors on different processes
2. Using Sentinel's remote dispatcher (future enhancement)
3. Centralizing message bus for inter-process communication

## Troubleshooting

### Issue: Skills not executing
- Check that runtime is started: `await bridge.start()`
- Verify skill is registered: `SubgraphRegistry.list_all()`
- Check logs for skill loading errors

### Issue: Retries not working
- Verify retry_policy is set: `bridge.retry_policy`
- Check that Governor is registered
- Monitor ExecutionFailed events via MessageBus

### Issue: State not synchronized
- Call `bridge.sync_to_session()` after MAF state updates
- Verify StateManager is passed to bridge
- Check that skills store results via StateManager

## References

- [Sentinel Framework Documentation](../sentinal/)
- [MAF Global Orchestrator](../global_/)
- [Skill Registry](./subgraph_registry.py)
- [Retry Policies](../sentinal/policies/)
"""
