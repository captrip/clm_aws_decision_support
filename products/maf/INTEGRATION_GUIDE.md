# Sentinel Integration Guide

## How to Add 3-Thread Supervision to Any MAF-Type Framework

This guide shows exactly what you need to integrate Sentinel into any MAF framework.

---

## Required Files from Sentinel Package

You need **only 1 file**:

```
sentinel_maf_integration.py
```

That's it! This file contains everything needed for 3-thread supervision.

---

## What's Inside `sentinel_maf_integration.py`

### Core Components:

1. **`SentinelExecutionState`** - Shared memory between threads
2. **`ActorThread`** - Executes the agent
3. **`ObserverThread`** - Watches memory changes
4. **`GovernorThread`** - Issues interventions
5. **`SentinelMAFAgent`** - Wrapper that runs any agent with 3 threads
6. **`wrap_maf_agent_with_sentinel()`** - Helper function to wrap agents

---

## Integration Steps

### Step 1: Copy the File

Copy `sentinel_maf_integration.py` to your framework:

```bash
cp sentinel_maf_integration.py /path/to/your/framework/
```

### Step 2: Import in Your Code

```python
from sentinel_maf_integration import wrap_maf_agent_with_sentinel
```

### Step 3: Wrap Your Agents

**Before (standard agent):**
```python
intent_agent = IntentUnderstandingAgent()
result = await intent_agent(state)
```

**After (supervised agent):**
```python
from sentinel_maf_integration import wrap_maf_agent_with_sentinel

# Wrap your agent
supervised_agent = wrap_maf_agent_with_sentinel(
    maf_agent=IntentUnderstandingAgent(),
    agent_name="Intent Understanding",
    quality_threshold=0.75,
)

# Use normally - 3 threads run automatically!
result = await supervised_agent(state)
```

Done! Your agent now runs with 3 threads.

---

## Integration Patterns

### Pattern 1: Wrap Individual Agents (Minimal Change)

**Use when:** You want to add supervision to specific agents only.

```python
from sentinel_maf_integration import wrap_maf_agent_with_sentinel

# Your existing agent
original_agent = YourAgent()

# Wrap it
supervised_agent = wrap_maf_agent_with_sentinel(
    maf_agent=original_agent,
    agent_name="Your Agent",
    quality_threshold=0.75,
)

# Use exactly like the original
result = await supervised_agent(input_state)
```

**Changes needed:** 
- ✅ Add 2 lines of code
- ✅ Replace agent variable

---

### Pattern 2: Wrap in Orchestrator (Framework-Wide)

**Use when:** You want all agents supervised automatically.

```python
from sentinel_maf_integration import SentinelMAFAgent

class YourOrchestrator:
    def __init__(self, enable_supervision=True):
        self.enable_supervision = enable_supervision
        
        # Create your agents
        intent_agent = IntentUnderstandingAgent()
        planner_agent = ActionPlannerAgent()
        
        # Wrap with supervision if enabled
        if self.enable_supervision:
            self.intent_agent = SentinelMAFAgent(
                maf_agent=intent_agent,
                agent_name="Intent Understanding",
                quality_threshold=0.75,
            )
            self.planner_agent = SentinelMAFAgent(
                maf_agent=planner_agent,
                agent_name="Action Planning",
                quality_threshold=0.70,
            )
        else:
            self.intent_agent = intent_agent
            self.planner_agent = planner_agent
    
    async def execute(self, state):
        # Use agents normally
        intent_result = await self.intent_agent(state)
        # ... rest of orchestration
```

**Changes needed:**
- ✅ Add conditional wrapping in `__init__`
- ✅ Add `enable_supervision` parameter

---

### Pattern 3: Factory Function (Reusable)

**Use when:** You want a clean, reusable setup.

```python
from sentinel_maf_integration import wrap_maf_agent_with_sentinel

def create_supervised_agents():
    """Factory to create all supervised agents."""
    return {
        "intent": wrap_maf_agent_with_sentinel(
            IntentUnderstandingAgent(),
            "Intent Understanding",
            quality_threshold=0.75,
        ),
        "planner": wrap_maf_agent_with_sentinel(
            ActionPlannerAgent(),
            "Action Planning",
            quality_threshold=0.70,
        ),
    }

# Use in orchestrator
agents = create_supervised_agents()
result = await agents["intent"](state)
```

**Changes needed:**
- ✅ Create factory function
- ✅ Call factory instead of direct agent creation

---

## What You Get

After integration, every supervised agent execution returns:

```python
{
    "execution_id": "exec-1-...",
    "status": "COMPLETED",
    "result": {...},  # Your agent's normal output
    
    # NEW: Supervision metadata
    "supervision": {
        "observer_observations": [
            {"timestamp": 1234.56, "status": "processing", ...},
            {"timestamp": 1235.67, "status": "complete", ...},
        ],
        "governor_interventions": [
            {"type": "quality_improvement", "quality": 0.5, ...}
        ],
    },
    
    # NEW: Runtime state
    "runtime_memory": {
        "actor_status": "complete",
        "latency_ms": 1234.56,
        "result": {...},
    },
    
    # NEW: Instructions from Governor
    "runtime_instructions": [
        "IMPROVE_QUALITY",
        "USE_BETTER_TOOLS"
    ],
}
```

---

## Configuration Options

### Quality Threshold

Controls when Governor intervenes:

```python
wrap_maf_agent_with_sentinel(
    maf_agent=agent,
    quality_threshold=0.9,  # Strict (intervene more often)
    quality_threshold=0.5,  # Lenient (intervene less)
)
```

### Timeout

How long to wait before timing out:

```python
SentinelMAFAgent(
    maf_agent=agent,
    timeout_sec=60.0,  # 60 seconds
)
```

### Enable/Disable Threads

Turn Observer or Governor on/off:

```python
SentinelMAFAgent(
    maf_agent=agent,
    enable_observer=True,   # Watch memory changes
    enable_governor=True,   # Monitor and intervene
)
```

---

## Dependencies

**Required:**
- Python 3.10+
- `asyncio` (standard library)
- `threading` (standard library)
- `dataclasses` (standard library)

**No external packages needed!** Everything uses Python standard library.

---

## Minimal Example

Here's the absolute minimum to add supervision:

```python
# 1. Import
from sentinel_maf_integration import wrap_maf_agent_with_sentinel

# 2. Wrap your agent
supervised = wrap_maf_agent_with_sentinel(
    maf_agent=YourAgent(),
    agent_name="Your Agent"
)

# 3. Use it
result = await supervised(input_state)

# 4. Check supervision
print(result['supervision'])  # Observer + Governor data
```

**That's it!** 3 OS threads now run automatically.

---

## Architecture Overview

```
Your Agent Call
      │
      ▼
┌─────────────────────────────────┐
│   SentinelMAFAgent Wrapper      │
└─────────────────────────────────┘
      │
      │ Creates shared state
      ▼
┌─────────────────────────────────┐
│  SentinelExecutionState         │
│  • runtime_memory               │
│  • runtime_instructions         │
└─────────────────────────────────┘
      │
      │ Starts 3 threads
      ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  ACTOR   │  │ OBSERVER │  │ GOVERNOR │
│  Thread  │  │  Thread  │  │  Thread  │
├──────────┤  ├──────────┤  ├──────────┤
│ Execute  │  │ Watch    │  │ Monitor  │
│ your     │  │ memory   │  │ quality  │
│ agent    │  │ changes  │  │          │
│          │  │          │  │ Issue    │
│ Update   │  │ Log      │  │ retries  │
│ memory   │  │ changes  │  │          │
└──────────┘  └──────────┘  └──────────┘
```

---

## Checklist for Integration

- [ ] Copy `sentinel_maf_integration.py` to your framework
- [ ] Import `wrap_maf_agent_with_sentinel` or `SentinelMAFAgent`
- [ ] Wrap your agents
- [ ] Test with a simple query
- [ ] Check `result['supervision']` for metadata
- [ ] (Optional) Add quality thresholds
- [ ] (Optional) Add to orchestrator
- [ ] (Optional) Make supervision configurable

---

## Example: Real Integration

**Your existing framework:**
```python
class MyFramework:
    def __init__(self):
        self.intent_agent = IntentAgent()
        self.planner = PlannerAgent()
    
    async def run(self, query):
        intent = await self.intent_agent({"query": query})
        plan = await self.planner(intent)
        return plan
```

**After integration:**
```python
from sentinel_maf_integration import wrap_maf_agent_with_sentinel

class MyFramework:
    def __init__(self, enable_supervision=True):
        # Wrap agents with supervision
        if enable_supervision:
            self.intent_agent = wrap_maf_agent_with_sentinel(
                IntentAgent(), "Intent Agent"
            )
            self.planner = wrap_maf_agent_with_sentinel(
                PlannerAgent(), "Planner"
            )
        else:
            self.intent_agent = IntentAgent()
            self.planner = PlannerAgent()
    
    async def run(self, query):
        # No changes needed here!
        intent = await self.intent_agent({"query": query})
        plan = await self.planner(intent)
        return plan

# Toggle supervision on/off
framework = MyFramework(enable_supervision=True)
```

**Changes made:**
1. Added import (1 line)
2. Added `enable_supervision` parameter (1 line)
3. Added conditional wrapping (6 lines)

**Total: 8 lines of code** to add 3-thread supervision! 🚀

---

## Summary

### What You Need:
- **1 file**: `sentinel_maf_integration.py`
- **1 import**: `wrap_maf_agent_with_sentinel`
- **2 lines**: Wrap your agent

### What You Get:
- ✅ 3 concurrent OS threads per agent
- ✅ Real-time memory observation
- ✅ Automatic quality monitoring
- ✅ Intervention capabilities
- ✅ Full supervision metadata
- ✅ Zero impact on existing code

### Compatibility:
- ✅ Works with any async agent
- ✅ Works with any agent framework
- ✅ No external dependencies
- ✅ Python 3.10+
