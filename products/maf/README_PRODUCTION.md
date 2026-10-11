# MAF + Sentinel Production Integration

## Overview

Your MAF framework now has **3-thread supervision** (Actor | Observer | Governor) built-in as a production feature.

## Core Files

### Production Files (Use These)

1. **`sentinel_maf_integration.py`** - Core integration framework
   - `SentinelExecutionState` - Shared memory between threads
   - `ActorThread` - Executes agents
   - `ObserverThread` - Watches memory changes
   - `GovernorThread` - Issues interventions
   - `SentinelMAFAgent` - Wrapper for any MAF agent

2. **`global_/sentinel_orchestrator.py`** - Production orchestrator
   - Drop-in replacement for `Global_Orchestrator`
   - All agents automatically supervised
   - Collects metrics across executions

3. **`test_production.py`** - Simple test file
   - Test with mock agents
   - Test with real MAF agents

### Archive (Reference Only)

All test/demo files moved to `archive/`:
- Test files
- Demo scripts
- Examples
- Visualizations

## Quick Start

### Option 1: Use Production Orchestrator

```python
from MAF.global_.sentinel_orchestrator import SentinelOrchestrator

# Create orchestrator with supervision
orc = SentinelOrchestrator(
    auto_load_skills=True,
    db_path="subgraph_registry.db",
    quality_threshold=0.75,
    enable_supervision=True,  # Enable 3-thread supervision
).build()

# Use normally - supervision happens automatically
result = await orc.ainvoke({
    "user_id": "user_123",
    "user_query": "What's the weather?",
})

# Check supervision metadata
print(result['supervision_metadata'])
```

### Option 2: Wrap Individual Agents

```python
from MAF.sentinel_maf_integration import wrap_maf_agent_with_sentinel
from MAF.MAFI.agents.IU_agents import IntentUnderstandingAgent

# Wrap your agent
supervised_agent = wrap_maf_agent_with_sentinel(
    maf_agent=IntentUnderstandingAgent(),
    agent_name="Intent Understanding",
    quality_threshold=0.75,
)

# Use normally - 3 threads run automatically
result = await supervised_agent({
    "user_query": "test query",
    "updated_query": "test query",
})
```

## Architecture

```
┌────────────────────────────────────┐
│   SentinelExecutionState (Shared)  │
│   • runtime_memory                 │
│   • runtime_instructions           │
└────────────────────────────────────┘
     │            │            │
Thread 1      Thread 2     Thread 3
     │            │            │
     ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│  ACTOR  │  │OBSERVER │  │GOVERNOR │
│         │  │         │  │         │
│ Execute │  │ Watch   │  │ Monitor │
│ agent   │  │ memory  │  │ quality │
│         │  │         │  │         │
│ Update  │  │ Log     │  │ Issue   │
│ memory  │  │ changes │  │ retries │
└─────────┘  └─────────┘  └─────────┘
```

## How It Works

### 1. Actor Thread
- Executes your MAF agent normally
- Updates `runtime_memory` as it runs
- Reads `runtime_instructions` from Governor
- Examples: "processing", "complete", "error"

### 2. Observer Thread
- Watches `runtime_memory` changes in real-time
- Logs state transitions
- Tracks metrics (latency, quality)
- Runs concurrently with Actor

### 3. Governor Thread
- Monitors execution quality
- Can issue instructions:
  - `RETRY_WITH_FALLBACK`
  - `IMPROVE_QUALITY`
  - `USE_BETTER_TOOLS`
  - `CHANGE_PARAMETERS`
- Runs concurrently with Actor and Observer

## Supervision Metadata

Every supervised execution returns:

```python
{
    "execution_id": "exec-1-...",
    "status": "COMPLETED",
    "result": {...},  # Your agent's output
    "supervision": {
        "observer_observations": [
            {"timestamp": ..., "status": "processing", ...},
            {"timestamp": ..., "status": "complete", ...},
        ],
        "governor_interventions": [
            {"type": "quality_improvement", "quality": 0.5, ...}
        ],
    },
    "runtime_memory": {...},
    "runtime_instructions": ["IMPROVE_QUALITY", ...],
}
```

## Configuration

### Enable/Disable Supervision

```python
# Enable supervision (default)
orc = SentinelOrchestrator(enable_supervision=True)

# Disable supervision (standard MAF)
orc = SentinelOrchestrator(enable_supervision=False)
```

### Quality Threshold

```python
# Strict quality checks
orc = SentinelOrchestrator(quality_threshold=0.9)

# Lenient quality checks
orc = SentinelOrchestrator(quality_threshold=0.5)
```

### Timeout

```python
supervised_agent = SentinelMAFAgent(
    maf_agent=your_agent,
    timeout_sec=60.0,  # Wait up to 60s
)
```

## Testing

### Test with Mock Agents

```bash
python test_production.py
```

This works without any LLM dependencies.

### Test with Real Agents

Requires:
- OpenAI API key in `.env`
- LangChain/LangGraph installed

```bash
python test_production.py
```

## Notebook Integration

Add to your Jupyter notebook:

```python
# Import
from MAF.global_.sentinel_orchestrator import SentinelOrchestrator

# Create
orc = SentinelOrchestrator(
    auto_load_skills=True,
    db_path="subgraph_registry.db",
).build()

# Use
result = await orc.ainvoke({
    "user_id": "notebook_user",
    "user_query": "Your query here",
})

# View supervision
print(result['supervision_metadata'])
```

## Metrics

Track orchestrator performance:

```python
metrics = orc.get_metrics()

print(f"Total executions: {metrics['execution_count']}")
print(f"Total observations: {metrics['total_observations']}")
print(f"Total interventions: {metrics['total_interventions']}")
```

## Troubleshooting

### "No module named 'langchain_core'"

Your notebook environment has LangChain, but the CLI doesn't.

**Solution**: Use the notebook (test.ipynb) to test with real agents.

### Supervision not working

Check:
```python
orc = SentinelOrchestrator(enable_supervision=True)  # Must be True
```

### Agent errors

Governor will detect errors and suggest retries automatically.

## Next Steps

1. **Test with mock agents**: `python test_production.py`
2. **Test in your notebook**: Add imports to test.ipynb
3. **Replace your orchestrator**: Use `SentinelOrchestrator` instead of `Global_Orchestrator`
4. **Monitor metrics**: Use `orc.get_metrics()` to track performance

## Summary

✅ **Production ready** - Clean, focused codebase  
✅ **3-thread architecture** - Actor | Observer | Governor  
✅ **Real OS threads** - True concurrent execution  
✅ **Drop-in compatible** - Works with existing MAF code  
✅ **Configurable** - Enable/disable supervision  
✅ **Metrics** - Track observations and interventions  
✅ **Quality monitoring** - Automatic intervention on low quality
