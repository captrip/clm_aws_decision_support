"""
Visual Demo of 3-Thread Architecture

Shows Actor, Observer, and Governor threads running concurrently
with real-time updates to the terminal.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import threading
import time
import asyncio
from typing import Any, Dict
from dataclasses import dataclass, field
from datetime import datetime, UTC


# ==========================================
# Visualization State
# ==========================================

class VisualState:
    """Tracks what each thread is doing for visualization."""

    def __init__(self):
        self.actor_status = "IDLE"
        self.observer_status = "IDLE"
        self.governor_status = "IDLE"

        self.actor_logs = []
        self.observer_logs = []
        self.governor_logs = []

        self.memory_updates = []
        self.instructions = []

        self.lock = threading.Lock()

    def update_actor(self, status: str, message: str = ""):
        with self.lock:
            self.actor_status = status
            if message:
                self.actor_logs.append(f"[{time.time():.2f}] {message}")

    def update_observer(self, status: str, message: str = ""):
        with self.lock:
            self.observer_status = status
            if message:
                self.observer_logs.append(f"[{time.time():.2f}] {message}")

    def update_governor(self, status: str, message: str = ""):
        with self.lock:
            self.governor_status = status
            if message:
                self.governor_logs.append(f"[{time.time():.2f}] {message}")

    def add_memory_update(self, key: str, value: Any):
        with self.lock:
            self.memory_updates.append((time.time(), key, str(value)[:50]))

    def add_instruction(self, instruction: str):
        with self.lock:
            self.instructions.append((time.time(), instruction))


# ==========================================
# Execution State
# ==========================================

@dataclass
class SentinelExecutionState:
    """Shared memory between threads."""

    execution_id: str
    runtime_memory: Dict[str, Any] = field(default_factory=dict)
    runtime_instructions: list = field(default_factory=list)
    status: str = "PENDING"
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_memory(self, key: str, value: Any):
        with self._lock:
            self.runtime_memory[key] = value

    def get_memory(self, key: str, default=None):
        with self._lock:
            return self.runtime_memory.get(key, default)

    def add_instruction(self, instruction: str):
        with self._lock:
            self.runtime_instructions.append(instruction)

    def get_instructions(self):
        with self._lock:
            return self.runtime_instructions.copy()


# ==========================================
# Visual Actor Thread
# ==========================================

class VisualActorThread(threading.Thread):
    """Actor with visualization."""

    def __init__(self, state: SentinelExecutionState, visual: VisualState, query: str):
        super().__init__(name="Actor", daemon=False)
        self.state = state
        self.visual = visual
        self.query = query

    def run(self):
        self.visual.update_actor("STARTING", "Actor thread started")
        time.sleep(0.2)

        # Step 1: Initialize
        self.visual.update_actor("INITIALIZING", "Initializing execution...")
        self.state.update_memory("actor_status", "initializing")
        self.visual.add_memory_update("actor_status", "initializing")
        time.sleep(0.3)

        # Step 2: Process query
        self.visual.update_actor("PROCESSING", f"Processing: {self.query[:30]}...")
        self.state.update_memory("actor_status", "processing")
        self.state.update_memory("current_query", self.query)
        self.visual.add_memory_update("actor_status", "processing")
        self.visual.add_memory_update("current_query", self.query[:30])
        time.sleep(0.5)

        # Step 3: Check instructions
        instructions = self.state.get_instructions()
        if instructions:
            self.visual.update_actor("FOLLOWING_INSTRUCTIONS",
                                   f"Following {len(instructions)} instructions")
            time.sleep(0.3)

        # Step 4: Execute (simulated)
        self.visual.update_actor("EXECUTING", "Executing MAF agent...")
        time.sleep(0.6)

        # Step 5: Generate result
        result = {
            "intent_understanding": f"Intent: {self.query[:50]}",
            "confidence": 0.85,
        }

        self.state.update_memory("result", result)
        self.state.update_memory("actor_status", "complete")
        self.state.update_memory("latency_ms", 450.0)

        self.visual.add_memory_update("result", str(result)[:40])
        self.visual.add_memory_update("actor_status", "complete")

        self.visual.update_actor("COMPLETE", "✓ Execution complete!")

        self.state.status = "COMPLETED"


# ==========================================
# Visual Observer Thread
# ==========================================

class VisualObserverThread(threading.Thread):
    """Observer with visualization."""

    def __init__(self, state: SentinelExecutionState, visual: VisualState, duration_sec: float):
        super().__init__(name="Observer", daemon=False)
        self.state = state
        self.visual = visual
        self.duration_sec = duration_sec
        self.running = True

    def run(self):
        self.visual.update_observer("STARTING", "Observer thread started")
        time.sleep(0.1)

        self.visual.update_observer("WATCHING", "Watching memory changes...")

        start_time = time.time()
        last_status = None
        observation_count = 0

        while self.running and (time.time() - start_time) < self.duration_sec:
            current_status = self.state.get_memory("actor_status")

            if current_status != last_status and current_status:
                observation_count += 1
                self.visual.update_observer("OBSERVED",
                                          f"Observed change #{observation_count}: {current_status}")
                last_status = current_status
                time.sleep(0.1)

            if self.state.status == "COMPLETED":
                self.visual.update_observer("COMPLETE",
                                          f"✓ Monitoring complete ({observation_count} observations)")
                break

            time.sleep(0.05)

        self.running = False


# ==========================================
# Visual Governor Thread
# ==========================================

class VisualGovernorThread(threading.Thread):
    """Governor with visualization."""

    def __init__(self, state: SentinelExecutionState, visual: VisualState,
                 duration_sec: float, quality_threshold: float):
        super().__init__(name="Governor", daemon=False)
        self.state = state
        self.visual = visual
        self.duration_sec = duration_sec
        self.quality_threshold = quality_threshold
        self.running = True

    def run(self):
        self.visual.update_governor("STARTING", "Governor thread started")
        time.sleep(0.15)

        self.visual.update_governor("MONITORING", "Monitoring execution...")

        start_time = time.time()
        intervention_count = 0

        while self.running and (time.time() - start_time) < self.duration_sec:
            actor_status = self.state.get_memory("actor_status")

            # Check for intervention opportunities
            if actor_status == "error":
                intervention_count += 1
                self.visual.update_governor("INTERVENING",
                                          f"⚠️  Issuing RETRY instruction")
                self.state.add_instruction("RETRY_WITH_FALLBACK")
                self.visual.add_instruction("RETRY_WITH_FALLBACK")
                time.sleep(0.2)

            elif actor_status == "complete":
                # Check quality
                result = self.state.get_memory("result", {})
                intent = result.get("intent_understanding", "")
                quality = 0.9 if len(str(intent)) > 20 else 0.3

                if quality < self.quality_threshold:
                    intervention_count += 1
                    self.visual.update_governor("INTERVENING",
                                              f"⚠️  Quality low ({quality:.2f}), issuing instructions")
                    self.state.add_instruction("IMPROVE_QUALITY")
                    self.state.add_instruction("USE_BETTER_TOOLS")
                    self.visual.add_instruction("IMPROVE_QUALITY")
                    self.visual.add_instruction("USE_BETTER_TOOLS")
                    time.sleep(0.2)

            if self.state.status == "COMPLETED":
                if intervention_count > 0:
                    self.visual.update_governor("COMPLETE",
                                              f"✓ Governance complete ({intervention_count} interventions)")
                else:
                    self.visual.update_governor("COMPLETE",
                                              "✓ No interventions needed")
                break

            time.sleep(0.1)

        self.running = False


# ==========================================
# Visualization Renderer
# ==========================================

def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def render_frame(visual: VisualState, frame_num: int):
    """Render a single frame of the visualization."""

    clear_screen()

    # Header
    print("=" * 80)
    print("  SENTINEL 3-THREAD ARCHITECTURE - LIVE VIEW")
    print("  Frame:", frame_num)
    print("=" * 80)
    print()

    # Thread Status
    print("┌─ THREAD STATUS " + "─" * 62 + "┐")
    print(f"│ 🎭 ACTOR:    [{visual.actor_status:^20}]                              │")
    print(f"│ 👁️  OBSERVER: [{visual.observer_status:^20}]                              │")
    print(f"│ 👑 GOVERNOR:  [{visual.governor_status:^20}]                              │")
    print("└" + "─" * 78 + "┘")
    print()

    # Shared Memory
    print("┌─ SHARED MEMORY (runtime_memory) " + "─" * 43 + "┐")
    if visual.memory_updates:
        recent = visual.memory_updates[-5:]  # Last 5 updates
        for _, key, value in recent:
            print(f"│  {key:20} = {value:50} │")
        if len(visual.memory_updates) > 5:
            print(f"│  ... ({len(visual.memory_updates) - 5} older updates)                                │")
    else:
        print("│  (empty)                                                             │")
    print("└" + "─" * 78 + "┘")
    print()

    # Instructions
    print("┌─ INSTRUCTIONS (from Governor to Actor) " + "─" * 37 + "┐")
    if visual.instructions:
        for _, instruction in visual.instructions:
            print(f"│  📜 {instruction:70} │")
    else:
        print("│  (none)                                                              │")
    print("└" + "─" * 78 + "┘")
    print()

    # Recent Logs
    print("┌─ RECENT ACTIVITY " + "─" * 59 + "┐")

    all_logs = []
    with visual.lock:
        if visual.actor_logs:
            all_logs.append(("🎭", visual.actor_logs[-1]))
        if visual.observer_logs:
            all_logs.append(("👁️ ", visual.observer_logs[-1]))
        if visual.governor_logs:
            all_logs.append(("👑", visual.governor_logs[-1]))

    if all_logs:
        for icon, log in all_logs[-3:]:  # Last 3
            print(f"│ {icon} {log:70} │")
    else:
        print("│  (no activity yet)                                                   │")

    print("└" + "─" * 78 + "┘")
    print()


# ==========================================
# Main Visual Demo
# ==========================================

def run_visual_demo():
    """Run the visual demo."""

    print("\n" + "=" * 80)
    print("  STARTING VISUAL DEMO")
    print("=" * 80)
    print("\n  Watch the 3 threads execute concurrently!")
    print("  Actor updates memory → Observer watches → Governor intervenes\n")
    print("  Press Ctrl+C to stop")
    print()

    time.sleep(2)

    # Create state
    state = SentinelExecutionState(execution_id="demo-001")
    visual = VisualState()

    # Create threads
    query = "How do I implement a supervised multi-agent system with real-time monitoring?"

    actor = VisualActorThread(state, visual, query)
    observer = VisualObserverThread(state, visual, duration_sec=5.0)
    governor = VisualGovernorThread(state, visual, duration_sec=5.0, quality_threshold=0.7)

    # Start threads
    actor.start()
    observer.start()
    governor.start()

    # Render loop
    frame_num = 0
    try:
        while actor.is_alive() or observer.is_alive() or governor.is_alive():
            render_frame(visual, frame_num)
            frame_num += 1
            time.sleep(0.1)

        # Final frame
        render_frame(visual, frame_num)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        visual.update_actor("STOPPED", "User interrupted")
        visual.update_observer("STOPPED", "User interrupted")
        visual.update_governor("STOPPED", "User interrupted")

    # Wait for threads
    actor.join(timeout=1)
    observer.join(timeout=1)
    governor.join(timeout=1)

    # Summary
    print("\n" + "=" * 80)
    print("  DEMO COMPLETE")
    print("=" * 80)
    print(f"\n  Total memory updates: {len(visual.memory_updates)}")
    print(f"  Total instructions issued: {len(visual.instructions)}")
    print(f"  Actor logs: {len(visual.actor_logs)}")
    print(f"  Observer logs: {len(visual.observer_logs)}")
    print(f"  Governor logs: {len(visual.governor_logs)}")
    print()
    print("  ✨ All 3 threads executed concurrently!")
    print()


# ==========================================
# Architecture Diagram
# ==========================================

def show_architecture_diagram():
    """Show ASCII diagram of the architecture."""

    print("\n" + "=" * 80)
    print("  SENTINEL 3-THREAD ARCHITECTURE")
    print("=" * 80)
    print()
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │                                                                     │")
    print("  │                     MAF Agent Execution                             │")
    print("  │                                                                     │")
    print("  └─────────────────────────────────────────────────────────────────────┘")
    print("                                    │")
    print("                                    │ User Query")
    print("                                    ▼")
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │                    SentinelExecutionState                           │")
    print("  │                    (Shared Memory)                                  │")
    print("  │                                                                     │")
    print("  │  • runtime_memory: {}      ← Actor writes, Observer/Governor read  │")
    print("  │  • runtime_instructions: [] ← Governor writes, Actor reads         │")
    print("  │  • status: PENDING → RUNNING → COMPLETED                           │")
    print("  │                                                                     │")
    print("  └─────────────────────────────────────────────────────────────────────┘")
    print("          │                    │                    │")
    print("          │                    │                    │")
    print("   Thread 1 ─────────   Thread 2 ─────────   Thread 3 ─────────")
    print("          │                    │                    │")
    print("          ▼                    ▼                    ▼")
    print("  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐")
    print("  │   🎭 ACTOR   │     │ 👁️  OBSERVER │     │ 👑 GOVERNOR  │")
    print("  ├──────────────┤     ├──────────────┤     ├──────────────┤")
    print("  │              │     │              │     │              │")
    print("  │ • Execute    │     │ • Watch      │     │ • Monitor    │")
    print("  │   MAF agent  │     │   memory     │     │   quality    │")
    print("  │              │     │   changes    │     │              │")
    print("  │ • Update     │     │              │     │ • Issue      │")
    print("  │   runtime    │     │ • Log        │     │   retries    │")
    print("  │   memory     │     │   changes    │     │              │")
    print("  │              │     │              │     │ • Suggest    │")
    print("  │ • Follow     │     │ • Detect     │     │   tools      │")
    print("  │   governor   │     │   anomalies  │     │              │")
    print("  │   instructions│    │              │     │ • Course     │")
    print("  │              │     │              │     │   correct    │")
    print("  │              │     │              │     │              │")
    print("  └──────────────┘     └──────────────┘     └──────────────┘")
    print()
    print("  Key Interactions:")
    print("  • Actor writes to runtime_memory → Observer sees changes in real-time")
    print("  • Governor monitors execution → Writes to runtime_instructions")
    print("  • Actor reads runtime_instructions → Follows Governor's guidance")
    print("  • All 3 threads run concurrently as separate OS threads")
    print()


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    import sys

    # Show diagram first
    show_architecture_diagram()

    input("Press Enter to start live demo...")

    # Run visual demo
    run_visual_demo()
