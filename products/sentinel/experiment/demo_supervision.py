"""
Supervision System Demo & Quick Start

Shows how to run and use the Agent + Observer + Governor system.
"""

import time
from agent_supervision import SupervisedAgent


# ==========================================
# Demo 1: Basic Supervision System
# ==========================================

def demo_basic_supervision():
    """
    Basic demonstration of supervised agent system.
    
    Shows:
    - Agent executing work
    - Observer watching in real-time
    - Governor making intervention decisions
    """
    
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC SUPERVISION SYSTEM")
    print("=" * 70)
    print("""
This demo shows:
  ✓ Agent Thread: Executes 10 steps of work
  ✓ Observer Thread: Watches agent behavior continuously
  ✓ Governor Thread: Analyzes and makes intervention decisions
  
Watch the console output to see all three threads working together.
You'll see:
  🤖 Agent messages (work execution)
  👁️  Observer messages (real-time monitoring)
  👑 Governor messages (analysis and interventions)
    """)
    
    input("Press Enter to start...")
    
    # Define work function
    def work_fn(step):
        """Simulate agent work."""
        # Step 2 is slow
        if step == 2:
            time.sleep(0.8)
        return {"step": step, "result": f"Completed step {step}"}
    
    # Run supervised agent
    supervised = SupervisedAgent(
        work_function=work_fn,
        total_steps=10,
    )
    
    supervised.run()


# ==========================================
# Demo 2: Custom Governance Rules
# ==========================================

def demo_custom_governance():
    """
    Shows how to customize governor behavior.
    """
    
    print("\n" + "=" * 70)
    print("DEMO 2: CUSTOM GOVERNANCE RULES")
    print("=" * 70)
    print("""
This demo shows how the governor can apply different rules:

ACTIVE RULES:
  1. Step 2 takes too long → Request SKIP
  2. Progress is steady → No intervention
  3. More than 7 steps → Suggest parameter modification
  4. Errors detected → ESCALATE for manual review

Watch how the governor identifies issues and requests corrections.
    """)
    
    input("Press Enter to start...")
    
    # Define work function
    def work_fn(step):
        """Simulate agent work with varying performance."""
        time.sleep(0.2)
        
        # Step 2 is slow
        if step == 2:
            time.sleep(0.6)
        
        # Step 8 has slow performance
        if step == 8:
            time.sleep(0.4)
        
        return {"step": step, "result": f"Completed step {step}"}
    
    # Run supervised agent
    supervised = SupervisedAgent(
        work_function=work_fn,
        total_steps=10,
    )
    
    supervised.run()


# ==========================================
# Demo 3: Agent with Error Handling
# ==========================================

def demo_error_handling():
    """
    Shows how supervision handles errors.
    """
    
    print("\n" + "=" * 70)
    print("DEMO 3: ERROR HANDLING AND ESCALATION")
    print("=" * 70)
    print("""
This demo shows:
  - Error detection
  - Escalation to manual review
  - Governor intervention on errors
  
Step 6 will trigger an error that the governor detects.
    """)
    
    input("Press Enter to start...")
    
    # Define work function that can fail
    def work_fn(step):
        """Simulate agent work with errors."""
        time.sleep(0.2)
        
        # Step 6 has an error
        if step == 6:
            raise RuntimeError("Step 6 failed - invalid data")
        
        return {"step": step, "result": f"Completed step {step}"}
    
    # Run supervised agent
    supervised = SupervisedAgent(
        work_function=work_fn,
        total_steps=10,
    )
    
    supervised.run()


# ==========================================
# Demo 4: Real-time Pause & Resume
# ==========================================

def demo_pause_resume():
    """
    Shows governor ability to pause and resume agent.
    """
    
    print("\n" + "=" * 70)
    print("DEMO 4: PAUSE & RESUME CONTROL")
    print("=" * 70)
    print("""
This demo shows:
  - Governor can pause agent execution
  - Agent resumes when signaled
  - Useful for dynamic throttling or rate limiting
    """)
    
    input("Press Enter to start...")
    
    def work_fn(step):
        """Simulate agent work."""
        time.sleep(0.3)
        return {"step": step, "result": f"Completed step {step}"}
    
    supervised = SupervisedAgent(
        work_function=work_fn,
        total_steps=10,
    )
    
    supervised.run()


# ==========================================
# Menu System
# ==========================================

def show_menu():
    """Display demo menu."""
    print("\n" + "=" * 70)
    print("AGENT SUPERVISION SYSTEM - DEMO MENU")
    print("=" * 70)
    print("""
This demonstrates the Agent + Observer + Governor supervision system:

  Agent Thread    → Executes work, checks for corrections
  Observer Thread → Watches behavior in real-time
  Governor Thread → Analyzes behavior, requests interventions

All three run concurrently!

OPTIONS:
  1. Basic Supervision Demo
  2. Custom Governance Rules Demo
  3. Error Handling Demo
  4. Pause & Resume Demo
  5. Exit
    """)


def main():
    """Main demo menu."""
    while True:
        show_menu()
        choice = input("Select demo (1-5): ").strip()
        
        if choice == "1":
            demo_basic_supervision()
        elif choice == "2":
            demo_custom_governance()
        elif choice == "3":
            demo_error_handling()
        elif choice == "4":
            demo_pause_resume()
        elif choice == "5":
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
