"""
Test Sentinel Integration Using Your Existing Notebook Setup

This uses the same setup as test.ipynb but adds Sentinel supervision.
"""

import sys
from pathlib import Path
_workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(_workspace_root))

import asyncio

# Import what you already have working in test.ipynb
# If this fails, it means we need to check your notebook imports

print("\n" + "=" * 70)
print("SENTINEL + YOUR MAF SETUP")
print("=" * 70)

print("\nAttempting to import your MAF components...")

try:
    from MAF.global_.orchestrator_graph import Global_Orchestrator
    print("✓ Global_Orchestrator imported")
except Exception as e:
    print(f"✗ Could not import Global_Orchestrator: {e}")

try:
    from MAF.global_.state import Orchestrator_Global_State
    print("✓ Orchestrator_Global_State imported")
except Exception as e:
    print(f"✗ Could not import Orchestrator_Global_State: {e}")

try:
    from MAF.MAFI.agent_factory.graph_factory_concurrent import SubgraphFactory
    print("✓ SubgraphFactory imported")
except Exception as e:
    print(f"✗ Could not import SubgraphFactory: {e}")

# Import Sentinel integration
try:
    from MAF.sentinel_maf_integration import wrap_maf_agent_with_sentinel
    print("✓ Sentinel integration imported")
except Exception as e:
    print(f"✗ Could not import Sentinel: {e}")

print("\n" + "=" * 70)
print("Next Steps:")
print("=" * 70)
print("""
To test Sentinel with your real MAF setup:

1. In your test.ipynb notebook, add this cell:

   from MAF.sentinel_maf_integration import wrap_maf_agent_with_sentinel

2. After building your orchestrator:

   orc = Global_Orchestrator(
       auto_load_skills=True,
       db_path="subgraph_registry.db",
       SubgraphFactory=SubgraphFactory,
   ).build()

3. Access the intent agent and wrap it:

   # Get the original intent agent from your orchestrator
   # (you'll need to find how it's accessed in your orchestrator)

   # Wrap with Sentinel
   from MAF.sentinel_maf_integration import SentinelMAFAgent

   # Example (adjust based on your orchestrator structure):
   # supervised_agent = wrap_maf_agent_with_sentinel(
   #     maf_agent=your_intent_agent,
   #     agent_name="Intent Understanding",
   #     quality_threshold=0.75,
   # )

4. Use normally:

   result = await supervised_agent({
       "user_query": "your query",
       "updated_query": "your query"
   })

This will give you 3-thread supervision (Actor|Observer|Governor)
for any agent in your orchestrator!
""")
