from __future__ import annotations

import asyncio

from MAF.global_.sentinel_orchestrator import SentinelOrchestrator


def test_mock_agent_runs_with_sentinel_supervision() -> None:
    class MockAgent:
        async def __call__(self, state: dict[str, str]) -> dict[str, str]:
            return {"intent_understanding": f"Intent: {state['user_query']}"}

    orchestrator = SentinelOrchestrator(enable_supervision=True)
    orchestrator.add_supervised_agent("Intent Understanding", MockAgent())

    result = asyncio.run(
        orchestrator.ainvoke(
            {"user_id": "test-user", "user_query": "Explain MONA routing."}
        )
    )

    assert result["intent_understanding"] == {
        "intent_understanding": "Intent: Explain MONA routing."
    }
    assert result["supervision_enabled"] is True
    assert result["supervision_metadata"] is not None
    assert result["orchestrator_stats"]["total_executions"] == 1
