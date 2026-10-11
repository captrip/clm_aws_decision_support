"""
AGENT + OBSERVER + GOVERNOR SUPERVISION SYSTEM

Comprehensive Documentation & Architecture Guide
"""

# ============================================================================
# 1. SYSTEM OVERVIEW
# ============================================================================

"""
ARCHITECTURE

┌─────────────────────────────────────────────────────────────────┐
│                     Main Execution Loop                          │
│                                                                  │
│  Three threads run in parallel, each with specific roles:        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ AGENT THREAD (User Work)                                 │  │
│  │ • Executes main agent logic                              │  │
│  │ • Makes decisions and calculations                       │  │
│  │ • Periodically checks for interventions from Governor   │  │
│  │ • Applies corrections as needed                          │  │
│  │ • Reports status and metrics                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ OBSERVER THREAD (Real-time Monitoring)                   │  │
│  │ • Continuously reads agent state                         │  │
│  │ • Creates observation snapshots                          │  │
│  │ • Publishes observations to queue                        │  │
│  │ • Tracks metrics and activity                            │  │
│  │ • Non-intrusive (read-only access)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ GOVERNOR THREAD (Decision Making & Intervention)         │  │
│  │ • Consumes observations from Observer                    │  │
│  │ • Analyzes agent behavior                                │  │
│  │ • Applies pre-defined governance rules                   │  │
│  │ • Queues intervention requests                           │  │
│  │ • Sets parameters, adjusts execution                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                           ↓↑
            Thread-Safe State Container
            (Shared via SupervisedAgentState)
                           ↓↑
    ┌────────────────────────────────────────────────┐
    │       Queues & Shared State                    │
    │                                                │
    │  • observation_queue: Observer → Governor     │
    │  • intervention_queue: Governor → Agent       │
    │  • agent_state: Current execution status     │
    │  • metrics: Performance data                  │
    │  • decisions: History of corrections          │
    │  • errors: Exception tracking                 │
    └────────────────────────────────────────────────┘

MESSAGE FLOW

Time →

Agent executes step 1
Agent reads: "Any corrections?" → No
Agent: Continue with step 2

Observer reads state → "Agent on step 2"
Observer: Create snapshot

Governor reads observation → "Step 2 detected"
Governor: Apply RULE 1 check
Governor: "Step 2 taking too long!"
Governor: Queue intervention (SKIP)

Agent reads: "Any corrections?" → YES (SKIP)
Agent: Skips step 2, moves to step 3

... process repeats ...


KEY DESIGN PRINCIPLES

1. Thread Safety
   - All shared state accessed via locks
   - Observer uses read-only access (no lock contention)
   - Governor writes intervention requests only
   - Agent is sole consumer of interventions

2. Non-Blocking Design
   - Observer doesn't block Agent
   - Governor doesn't block Agent
   - Agent checks for corrections at safe points
   - No unexpected interruptions

3. Observable & Debuggable
   - Every decision is logged
   - Metrics track behavior
   - Observation snapshots capture state
   - Complete execution history available

4. Extensible Governance
   - Add new rules to Governor
   - Custom intervention types
   - Pluggable decision logic
   - Easy to test different policies
"""


# ============================================================================
# 2. COMPONENTS DETAILED
# ============================================================================

"""
2.1 AGENT THREAD
================

Responsibilities:
  ✓ Execute main work function (user-provided)
  ✓ Maintain execution state and progress
  ✓ Regularly check for interventions
  ✓ Apply corrections from Governor
  ✓ Update metrics and decisions
  ✓ Handle errors gracefully

Execution Loop:
  for each step:
    1. Check for pending interventions
    2. If intervention exists:
       a. Process intervention type (RETRY, SKIP, MODIFY_PARAMS, etc.)
       b. Update state accordingly
       c. Log decision
       d. Continue or skip based on intervention
    3. Update current_operation in state
    4. Execute work_function(step)
    5. Store results in metrics
    6. Wait briefly (allow other threads to run)
    7. Repeat

Intervention Handling:
  RETRY         → Re-execute same step
  SKIP          → Skip to next step
  MODIFY_PARAMS → Update execution parameters
  PAUSE         → Temporarily pause (resume on signal)
  HALT          → Stop execution immediately
  ESCALATE      → Pause and request manual review

Thread Safety:
  - All state reads/writes use SupervisedAgentState.get/update
  - Uses locks internally (transparent to agent code)
  - No risk of data races


2.2 OBSERVER THREAD
===================

Responsibilities:
  ✓ Continuously monitor agent state
  ✓ Create observation snapshots
  ✓ Publish observations for Governor
  ✓ Track metrics over time
  ✓ Detect anomalies (optional)

Observation Cycle:
  while running:
    1. Read agent state (snapshot)
    2. Create ObservationSnapshot:
       - timestamp
       - status
       - current_operation
       - progress
       - metrics (copy)
       - decisions (copy)
    3. Push to observation_queue
    4. Display observation (optional)
    5. Sleep 0.5 seconds
    6. Repeat

Key Features:
  - Non-intrusive (read-only)
  - No locks needed (reads snapshot)
  - Can run frequently without overhead
  - Historical data available

Output:
  Observer messages like:
    👁️ Observer: [42] working | Processing Step 5 | 5/10 steps
    👁️ Observer: [43] working | Processing Step 6 | 6/10 steps


2.3 GOVERNOR THREAD
===================

Responsibilities:
  ✓ Analyze observations from Observer
  ✓ Apply governance rules to behavior
  ✓ Make intervention decisions
  ✓ Queue corrections for Agent
  ✓ Track interventions and rationale

Decision Cycle:
  while running:
    1. Pop observation from queue
    2. If observation exists:
       a. Apply governance rules
       b. Check each rule condition
       c. If rule triggered:
          - Create intervention request
          - Queue for Agent
          - Log decision
       d. Increment analysis counter
    3. Sleep 0.3 seconds
    4. Repeat

Governance Rules:
  Rules are applied to each observation. Examples:

  RULE 1: Step Detection
    if observation.progress == 2:
      trigger SKIP intervention
      reason: "Step 2 taking too long"

  RULE 2: Progress Monitoring
    if steps_completed > 2:
      no intervention
      reason: "Progress is steady"

  RULE 3: Threshold Detection
    if observation.progress > 7:
      trigger MODIFY_PARAMS intervention
      reason: "Many steps executed"

  RULE 4: Error Escalation
    if len(errors) > 0 and progress > 5:
      trigger ESCALATE intervention
      reason: f"Errors detected: {errors}"

To Add Custom Rules:
  1. Modify _analyze_and_intervene() method
  2. Add new if/elif conditions
  3. Call _request_intervention() when triggered
  4. Set appropriate priority (1-5)

Intervention Priorities:
  1 = Low (suggestions, optimizations)
  2 = Normal (adjustments)
  3 = Medium (corrective actions)
  4 = High (major corrections)
  5 = Critical (halt, escalate, emergency)
"""


# ============================================================================
# 3. USAGE GUIDE
# ============================================================================

"""
3.1 BASIC USAGE
===============

```python
from agent_supervision import SupervisedAgent

# Define work function
def my_work_function(step):
    # Do work for this step
    result = step * 10  # Example
    return result

# Create supervised agent
supervised = SupervisedAgent(
    work_function=my_work_function,
    total_steps=10,
)

# Run with supervision
supervised.run()

# Access results
snapshot = supervised.state.get_snapshot()
print(f"Final status: {snapshot['status']}")
print(f"Decisions made: {snapshot['decisions']}")
print(f"Metrics: {snapshot['metrics']}")
```


3.2 CUSTOM GOVERNOR RULES
=========================

To add custom governance rules, modify GovernorThread._analyze_and_intervene():

```python
def _analyze_and_intervene(self, observation):
    # Example: Custom rule for performance monitoring
    
    if observation.progress > 5:
        # Get execution time
        elapsed = time.time() - observation.timestamp
        
        # If taking too long, request adjustment
        if elapsed > 10:
            self._request_intervention(
                CorrectionType.MODIFY_PARAMS,
                "Execution exceeding time budget",
                parameters={"timeout": 20},
                priority=3,
            )
```


3.3 WITH MAF AGENTS
===================

```python
from maf_supervision import SupervisedMAFAgent

# Define your MAF agent function
async def my_maf_agent():
    # Your agent logic here
    result = await orchestrator.execute_workflow(...)
    return result

# Create supervised wrapper
supervised = SupervisedMAFAgent(
    agent_name="MyAgent",
    agent_execute_fn=my_maf_agent,
    state_manager=state_manager,
)

# Run with supervision
result = await supervised.run_supervised()
```


3.4 MULTI-AGENT SUPERVISION
===========================

```python
from maf_supervision import SupervisedMultiAgentSystem

# Create system
system = SupervisedMultiAgentSystem()

# Register agents
system.register_agent("SearchAgent", search_agent_fn)
system.register_agent("AnalysisAgent", analysis_agent_fn)
system.register_agent("SummaryAgent", summary_agent_fn)

# Run all with supervision
results = await system.run_all_supervised()
```
"""


# ============================================================================
# 4. STATE MANAGEMENT
# ============================================================================

"""
4.1 SUPERVISED AGENT STATE
===========================

The SupervisedAgentState class manages all shared state:

Attributes:
  status: Current execution status (IDLE, WORKING, PAUSED, etc.)
  current_operation: What agent is currently doing
  progress: Steps completed so far
  total_steps: Total steps expected
  metrics: Dict of performance metrics
  decisions: List of decisions/corrections applied
  errors: List of errors encountered
  observation_queue: Queue of observations for Governor
  intervention_queue: Queue of interventions for Agent

Thread-Safe Methods:
  update(key, value)              - Set a state value
  get(key, default)               - Read a state value
  get_snapshot()                  - Get complete state copy
  push_observation(obs)           - Add observation
  pop_observation()               - Get oldest observation
  push_intervention(intervention) - Add intervention
  pop_intervention()              - Get highest priority intervention
  has_pending_interventions()     - Check for interventions
  add_metric(key, value)          - Add performance metric
  add_decision(decision)          - Log a decision
  add_error(error)                - Log an error


4.2 METRICS & MONITORING
=========================

Available Metrics:
  step_N_result: Result of step N
  step_N_time: Execution time of step N
  param_modifications: Parameter changes applied
  agent_result: Final agent output
  total_execution_time: Total time elapsed
  intervention_count: Number of interventions
  decision_count: Number of decisions made

Access Metrics:
  snapshot = state.get_snapshot()
  metrics = snapshot['metrics']
  print(metrics)  # {'step_1_time': 0.23, 'step_2_time': 0.54, ...}


4.3 DECISIONS & AUDIT TRAIL
============================

All decisions are logged:
  - What intervention was applied
  - Why (reason from Governor)
  - When (timestamp)
  - Result (success, skip, retry, etc.)

Access Audit Trail:
  snapshot = state.get_snapshot()
  decisions = snapshot['decisions']
  for decision in decisions:
    print(decision)  # "Skip step 2", "Modify params: {...}", etc.
"""


# ============================================================================
# 5. THREAD SAFETY & CONCURRENCY
# ============================================================================

"""
5.1 THREAD COORDINATION
=======================

Three threads coordinate via:
  1. observation_queue: Observer → Governor
  2. intervention_queue: Governor → Agent
  3. agent_state dict: Shared access via locks

No Race Conditions Because:
  - observation_queue is append-only from Observer
  - intervention_queue is append-only from Governor
  - agent_state uses threading.Lock() for all access
  - Agent is sole consumer of interventions
  - Observer only reads, never writes state


5.2 LOCK GRANULARITY
====================

All state access goes through SupervisedAgentState:
  - Single lock protects entire state dict
  - No fine-grained locks needed (not a bottleneck)
  - Observer reads are fast (snapshot copy)
  - Governor writes are infrequent
  - Agent accesses are regular but not performance-critical


5.3 AVOIDING DEADLOCKS
======================

Deadlock Prevention:
  ✓ Single lock only (no multiple locks to deadlock)
  ✓ No nested lock acquisition
  ✓ Locks held for minimal time
  ✓ No circular wait conditions


5.4 PERFORMANCE IMPLICATIONS
============================

Overhead Analysis:
  - Lock operations: ~1 microsecond
  - Observation snapshot: ~1 millisecond
  - State update: ~0.1 millisecond
  - Thread context switch: ~10 microseconds

For typical agent with 10 steps:
  - Total supervision overhead: < 5 milliseconds
  - Negligible compared to actual work
  - Worth it for intervention capability
"""


# ============================================================================
# 6. INTERVENTIONS & ERROR HANDLING
# ============================================================================

"""
6.1 INTERVENTION TYPES
======================

CorrectionType Enum:
  RETRY:          Re-execute current step
  SKIP:           Skip to next step
  MODIFY_PARAMS:  Adjust execution parameters
  PAUSE:          Temporarily pause
  RESUME:         Resume from pause
  HALT:           Stop execution immediately
  ESCALATE:       Pause for manual review

Each intervention includes:
  - correction_type: What to do
  - reason: Why governor requested it
  - parameters: Optional dict with details
  - priority: 1-5 (higher priority first)


6.2 ERROR HANDLING
==================

Errors are caught and logged:
  - Agent errors: Caught in work_function
  - Governor errors: Caught in rule evaluation
  - State errors: Caught in lock acquisition
  - Thread errors: Caught at thread level

All errors stored in state.errors for audit trail.


6.3 ESCALATION PROCESS
======================

When to escalate:
  1. Multiple errors detected
  2. Agent stuck for too long
  3. Unknown/unexpected state
  4. Resource constraints violated

Escalation Flow:
  Governor: "Error detected!"
  Governor: Queue ESCALATE intervention
  Agent: "I'm escalated, pausing..."
  Agent: Update status to PAUSED
  Human/System: Review error and decide next action


6.4 RECOVERY STRATEGIES
=======================

Retry with Backoff:
  if attempt < 3:
    Governor: Request RETRY
    Agent: Re-execute (automatic)
    if success:
      Continue
    else:
      Wait, increment attempt counter

Graceful Degradation:
  if error is recoverable:
    Governor: Request SKIP with context
    Agent: Skip problematic step
    Continue with remaining steps

Manual Intervention:
  if error is critical:
    Governor: Request ESCALATE
    System: Pause and alert operator
    Operator: Review and decide
"""


# ============================================================================
# 7. DEPLOYMENT & INTEGRATION
# ============================================================================

"""
7.1 INTEGRATION WITH MAF
=========================

The SupervisedMAFAgent wraps MAF agent functions:

Benefits:
  ✓ Monitor LLM agent behavior
  ✓ Detect hallucinations or errors
  ✓ Enforce quality constraints
  ✓ Apply parameter adjustments
  ✓ Real-time intervention capability

Example:
  supervised = SupervisedMAFAgent(
    agent_name="RAGAgent",
    agent_execute_fn=rag_agent_function,
  )
  result = await supervised.run_supervised()


7.2 INTEGRATION WITH SENTINEL
==============================

Future Enhancement - Supervised Agents with Sentinel:
  - Wrap SupervisedAgent as Sentinel Executable
  - Use Sentinel Governor for high-level orchestration
  - Run supervised agents within Sentinel actor system
  - Combine thread-based local supervision with actor-based distribution


7.3 PRODUCTION DEPLOYMENT
==========================

Considerations:
  1. Logging: Capture all decisions and interventions
  2. Monitoring: Track observation counts and intervention rates
  3. Metrics: Store performance data for analysis
  4. Alerting: Alert on repeated escalations
  5. Tuning: Adjust governance rules based on behavior
  6. Scaling: Run multiple supervised agents with thread pooling


7.4 CONFIGURATION
=================

Key Configuration Points:

Agent Parameters:
  - total_steps: How many steps expected
  - work_function: Function to execute
  - timeout: Max time per step

Observer Parameters:
  - observation_interval: How often to observe (default 0.5s)
  - buffer_size: Max observations in queue

Governor Parameters:
  - governance_rules: Rules to apply
  - analysis_interval: How often to analyze (default 0.3s)
  - priority_threshold: Min priority to act on (default 1)
"""


# ============================================================================
# 8. TROUBLESHOOTING
# ============================================================================

"""
8.1 COMMON ISSUES
=================

Issue: Agent seems stuck
  Solution: Governor escalates after timeout
  Action: Review error messages in state.errors

Issue: Too many interventions
  Solution: Adjust governance rules
  Action: Modify Governor._analyze_and_intervene()

Issue: Interventions not taking effect
  Solution: Agent might not check for them
  Action: Ensure Agent checks pop_intervention() regularly

Issue: High CPU usage
  Solution: Adjust thread sleep intervals
  Action: Increase observation_interval, analysis_interval

Issue: State inconsistency
  Solution: Lock conflict
  Action: Check for nested lock acquisitions (shouldn't happen)


8.2 DEBUGGING
=============

Enable verbose logging:
  1. Capture all print() outputs
  2. Store to file for analysis
  3. Review decision sequence
  4. Trace intervention flow

Debug State:
  snapshot = state.get_snapshot()
  print(f"Status: {snapshot['status']}")
  print(f"Progress: {snapshot['progress']}/{snapshot['total_steps']}")
  print(f"Decisions: {snapshot['decisions']}")
  print(f"Errors: {snapshot['errors']}")


8.3 PERFORMANCE ANALYSIS
========================

Metrics to Monitor:
  - Observation count (from observer)
  - Intervention count (from governor)
  - Decision count (from agent)
  - Total execution time
  - Time per step
  - Error count

Performance Targets:
  - Observation overhead: < 1%
  - Governor overhead: < 1%
  - Total supervision overhead: < 5%
"""


if __name__ == "__main__":
    print(__doc__)
    print("\nFor detailed examples, see demo_supervision.py")
