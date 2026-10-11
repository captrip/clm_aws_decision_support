"""
Simple Visual Demo - Auto-running
Shows the 3 threads in action
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import threading
import time
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class State:
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


class Actor(threading.Thread):
    def __init__(self, state: State):
        super().__init__(name="Actor")
        self.state = state

    def run(self):
        print("🎭 [ACTOR] Starting...")
        time.sleep(0.2)

        print("🎭 [ACTOR] Initializing...")
        self.state.update_memory("actor_status", "initializing")
        time.sleep(0.3)

        print("🎭 [ACTOR] Processing query...")
        self.state.update_memory("actor_status", "processing")
        self.state.update_memory("query", "Build supervised agent system")
        time.sleep(0.5)

        instructions = self.state.runtime_instructions
        if instructions:
            print(f"🎭 [ACTOR] Following {len(instructions)} instructions from Governor")
            time.sleep(0.2)

        print("🎭 [ACTOR] Executing MAF agent...")
        self.state.update_memory("actor_status", "executing")
        time.sleep(0.6)

        print("🎭 [ACTOR] Generating result...")
        result = {"intent_understanding": "Intent: Build supervised agent system", "confidence": 0.85}
        self.state.update_memory("result", result)
        self.state.update_memory("actor_status", "complete")
        self.state.update_memory("latency_ms", 450.0)

        print("🎭 [ACTOR] ✓ Complete!")
        self.state.status = "COMPLETED"


class Observer(threading.Thread):
    def __init__(self, state: State):
        super().__init__(name="Observer")
        self.state = state
        self.observations = 0

    def run(self):
        print("👁️  [OBSERVER] Starting...")
        time.sleep(0.1)

        print("👁️  [OBSERVER] Watching memory...")

        last_status = None
        while self.state.status != "COMPLETED":
            current = self.state.get_memory("actor_status")

            if current != last_status and current:
                self.observations += 1
                print(f"👁️  [OBSERVER] Observed change #{self.observations}: actor_status = {current}")
                last_status = current
                time.sleep(0.1)

            time.sleep(0.05)

        print(f"👁️  [OBSERVER] ✓ Complete ({self.observations} observations)")


class Governor(threading.Thread):
    def __init__(self, state: State):
        super().__init__(name="Governor")
        self.state = state
        self.interventions = 0

    def run(self):
        print("👑 [GOVERNOR] Starting...")
        time.sleep(0.15)

        print("👑 [GOVERNOR] Monitoring execution...")

        while self.state.status != "COMPLETED":
            actor_status = self.state.get_memory("actor_status")

            if actor_status == "error":
                self.interventions += 1
                print("👑 [GOVERNOR] ⚠️  Detected error! Issuing RETRY instruction")
                self.state.add_instruction("RETRY_WITH_FALLBACK")
                time.sleep(0.2)

            elif actor_status == "complete":
                result = self.state.get_memory("result", {})
                intent = result.get("intent_understanding", "")
                quality = 0.9 if len(str(intent)) > 20 else 0.3

                if quality < 0.7:
                    self.interventions += 1
                    print(f"👑 [GOVERNOR] ⚠️  Quality low ({quality:.2f})! Issuing IMPROVE_QUALITY")
                    self.state.add_instruction("IMPROVE_QUALITY")
                    self.state.add_instruction("USE_BETTER_TOOLS")
                    time.sleep(0.2)

            time.sleep(0.1)

        if self.interventions > 0:
            print(f"👑 [GOVERNOR] ✓ Complete ({self.interventions} interventions)")
        else:
            print("👑 [GOVERNOR] ✓ Complete (no interventions needed)")


def show_diagram():
    print("\n" + "=" * 80)
    print("SENTINEL 3-THREAD ARCHITECTURE")
    print("=" * 80)
    print()
    print("                        MAF Agent Execution")
    print("                                │")
    print("                                ▼")
    print("           ┌────────────────────────────────────────┐")
    print("           │   SentinelExecutionState (Shared)      │")
    print("           │   • runtime_memory                     │")
    print("           │   • runtime_instructions               │")
    print("           └────────────────────────────────────────┘")
    print("                 │              │              │")
    print("         Thread 1 ───    Thread 2 ───    Thread 3 ───")
    print("                 │              │              │")
    print("                 ▼              ▼              ▼")
    print("           ┌─────────┐    ┌─────────┐    ┌─────────┐")
    print("           │  🎭 ACTOR│    │👁️ OBSERVER│   │👑 GOVERNOR│")
    print("           ├─────────┤    ├─────────┤    ├─────────┤")
    print("           │ Execute │    │ Watch   │    │ Monitor │")
    print("           │ agent   │    │ memory  │    │ quality │")
    print("           │         │    │         │    │         │")
    print("           │ Update  │    │ Log     │    │ Issue   │")
    print("           │ memory  │    │ changes │    │ retries │")
    print("           └─────────┘    └─────────┘    └─────────┘")
    print()
    print("  Flow:")
    print("  1. Actor executes & writes to runtime_memory")
    print("  2. Observer watches memory changes in real-time")
    print("  3. Governor monitors & writes runtime_instructions")
    print("  4. Actor reads instructions & follows them")
    print()


def run_demo():
    print("\n" + "=" * 80)
    print("LIVE DEMO - 3 Threads Running Concurrently")
    print("=" * 80)
    print()

    # Create shared state
    state = State()

    # Create threads
    actor = Actor(state)
    observer = Observer(state)
    governor = Governor(state)

    print("Starting all 3 threads...\n")

    # Start all threads
    actor.start()
    observer.start()
    governor.start()

    # Wait for completion
    actor.join()
    observer.join()
    governor.join()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

    print(f"\nFinal State:")
    print(f"  Status: {state.status}")
    print(f"  Memory keys: {list(state.runtime_memory.keys())}")
    print(f"  Instructions issued: {len(state.runtime_instructions)}")

    if state.runtime_instructions:
        print(f"\n  Instructions from Governor:")
        for inst in state.runtime_instructions:
            print(f"    - {inst}")

    print(f"\n  Thread IDs:")
    print(f"    Actor: {actor.ident}")
    print(f"    Observer: {observer.ident}")
    print(f"    Governor: {governor.ident}")

    print("\n  ✨ All 3 OS threads executed concurrently!")
    print()


if __name__ == "__main__":
    show_diagram()
    run_demo()
