"""
MAF Supervised Agent Demo with Real User Queries

Interactive demo that accepts real user queries and processes them
through a supervised intent understanding agent.
"""

import sys
from pathlib import Path

# Add workspace root to path for imports
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio
import time
from typing import Any, Dict

from experiment.agent_supervision import (
    SupervisedAgentState,
    AgentStatus,
)
from MAF.supervised_subagents import SubagentObserver, SubagentGovernor


# Mock MAF Agent (simulates IntentUnderstandingAgent)
class MockIntentUnderstandingAgent:
    """Mock MAF agent that processes intent."""

    def __init__(self, name: str = "Intent Understanding Agent"):
        self.description = "Intent Understanding"
        self.name = name
        self.call_count = 0

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user query and extract intent."""
        self.call_count += 1
        await asyncio.sleep(0.2)  # Simulate processing time

        user_query = state.get("user_query", "")

        # Simple intent extraction (in real MAF this would use LLM)
        intent_info = self._extract_intent(user_query)

        return {
            "intent_understanding": intent_info,
            "processed_query": user_query,
            "confidence": 0.85,
        }

    def _extract_intent(self, query: str) -> str:
        """Simple intent extraction based on keywords."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["weather", "temperature", "forecast"]):
            return "Intent: Weather Information Request | Domain: Weather | Action: Get current conditions"
        elif any(word in query_lower for word in ["joke", "funny", "laugh"]):
            return "Intent: Entertainment Request | Domain: Humor | Action: Generate joke"
        elif any(word in query_lower for word in ["time", "date", "clock"]):
            return "Intent: Time Information Request | Domain: Time | Action: Get current time/date"
        elif any(word in query_lower for word in ["hello", "hi", "hey"]):
            return "Intent: Greeting | Domain: Social | Action: Respond with greeting"
        elif any(word in query_lower for word in ["help", "assist", "support"]):
            return "Intent: Help Request | Domain: Support | Action: Provide assistance"
        elif any(word in query_lower for word in ["search", "find", "look"]):
            return "Intent: Search Request | Domain: Information Retrieval | Action: Perform search"
        else:
            return f"Intent: General Query | Domain: General | Action: Process query | Query: {query[:100]}"


# Supervised wrapper
class SupervisedIntentUnderstandingAgent:
    """Wraps MAF IntentUnderstandingAgent with full supervision."""

    def __init__(self, original_agent: Any = None):
        if original_agent is None:
            self.original_agent = MockIntentUnderstandingAgent()
        else:
            self.original_agent = original_agent

        self.name = f"Supervised-{self.original_agent.name}"
        self.description = f"Supervised {self.original_agent.description}"

        # Supervision state
        self.state = SupervisedAgentState()
        self.observer = None
        self.governor = None

        # Metrics
        self.execution_count = 0
        self.intervention_count = 0

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with supervision."""
        self.execution_count += 1

        self.state.update("status", AgentStatus.WORKING)
        self.state.update("current_operation", "Intent Understanding")

        # Start observer and governor
        self.observer = SubagentObserver(self.state, "intent")
        self.governor = SubagentGovernor(self.state, "intent")

        self.observer.start()
        self.governor.start()

        try:
            # Execute original agent
            start_time = time.time()
            result = await self.original_agent(state)
            latency_ms = (time.time() - start_time) * 1000

            # Compute quality
            quality_score = self._compute_quality(result)

            # Update metrics
            self.state.add_metric("latency_ms", latency_ms)
            self.state.add_metric("quality_score", quality_score)

            self.state.update("status", AgentStatus.COMPLETE)

            print(f"  ✅ Quality: {quality_score:.2f} | Latency: {latency_ms:.0f}ms")

            return result

        except Exception as e:
            self.state.update("status", AgentStatus.ERROR)
            self.state.add_error(str(e))
            print(f"  ❌ Error: {e}")
            raise

        finally:
            # Stop monitoring
            if self.observer:
                self.observer.stop()
            if self.governor:
                self.governor.stop()
            if self.observer:
                self.observer.join(timeout=1)
            if self.governor:
                self.governor.join(timeout=1)

    def _compute_quality(self, result: Dict[str, Any]) -> float:
        """Compute quality score based on result completeness."""
        score = 0.0

        # Check if intent_understanding exists and is non-empty
        intent_text = result.get("intent_understanding", "")
        if intent_text and len(intent_text.strip()) > 10:
            score += 0.5

        # Check if confidence exists
        if "confidence" in result:
            score += 0.3

        # Check if processed_query exists
        if "processed_query" in result:
            score += 0.2

        return min(score, 1.0)


# Interactive demo
async def interactive_demo():
    """Run interactive demo with user input."""
    print("\n" + "=" * 70)
    print("MAF Supervised Agent - Interactive Demo")
    print("=" * 70)
    print("\nThis demo shows a supervised intent understanding agent processing")
    print("real user queries with monitoring and quality metrics.")
    print("\nType 'quit' or 'exit' to stop.")
    print("=" * 70)

    # Create supervised agent
    agent = SupervisedIntentUnderstandingAgent()

    print(f"\n✨ Agent initialized: {agent.name}")
    print(f"📝 Description: {agent.description}\n")

    while True:
        try:
            # Get user input
            print("-" * 70)
            user_query = input("\n💬 Your query: ").strip()

            if not user_query:
                continue

            if user_query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            # Process query
            print(f"\n🚀 Processing with supervision...")

            state = {
                "user_query": user_query,
                "updated_query": user_query,
                "user_id": "demo_user",
                "timestamp": time.time(),
            }

            result = await agent(state)

            # Display results
            print(f"\n📤 Results:")
            print(f"  Intent: {result.get('intent_understanding', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")

            print(f"\n📊 Session Stats:")
            print(f"  Total queries processed: {agent.execution_count}")
            print(f"  Interventions: {agent.intervention_count}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


async def batch_demo():
    """Run batch demo with predefined queries."""
    print("\n" + "=" * 70)
    print("MAF Supervised Agent - Batch Demo")
    print("=" * 70)

    agent = SupervisedIntentUnderstandingAgent()

    queries = [
        "What's the weather like today?",
        "Tell me a funny joke",
        "What time is it?",
        "Help me find a restaurant nearby",
        "Search for machine learning tutorials",
        "Hello! How are you?",
    ]

    print(f"\n🚀 Processing {len(queries)} queries...\n")

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: {query}")

        state = {
            "user_query": query,
            "updated_query": query,
            "user_id": "batch_user",
        }

        result = await agent(state)
        print(f"  Intent: {result['intent_understanding']}")

    print(f"\n✅ Batch complete!")
    print(f"📊 Total processed: {agent.execution_count}")


async def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        await batch_demo()
    else:
        await interactive_demo()


if __name__ == "__main__":
    asyncio.run(main())
