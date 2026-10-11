from __future__ import annotations

import asyncio

from MAF.integration import MAFSentinelRuntimeBridge
from MAFI.agent_factory.types.state_manager import StateManager
from sentinal.policies.exponential import ExponentialRetryPolicy


class MockGraph:
    async def ainvoke(
        self,
        state: dict[str, object],
        config: dict[str, object],
    ) -> dict[str, object]:
        return {"summary": state["user_query"]}


class MockOrchestrator:
    def __init__(self) -> None:
        self.state_manager = StateManager()
        self.graph = MockGraph()


def test_runtime_bridge_executes_workflow_under_sentinel() -> None:
    orchestrator = MockOrchestrator()
    bridge = MAFSentinelRuntimeBridge(orchestrator, max_concurrent_skills=1)

    async def execute() -> dict[str, object]:
        await bridge.start()
        try:
            return await bridge.execute_workflow("Test integrated execution")
        finally:
            await bridge.stop()

    result = asyncio.run(execute())

    assert result["response"] == "Test integrated execution"
    assert result["status"] == "completed"
    assert orchestrator.state_manager.get_scope("sentinel")["status"] == "completed"


def test_runtime_bridge_executes_individual_skill() -> None:
    orchestrator = MockOrchestrator()
    bridge = MAFSentinelRuntimeBridge(orchestrator)

    async def runner(state: dict[str, object]) -> dict[str, object]:
        return {"answer": state["query"]}

    result = asyncio.run(
        bridge.execute_skill(
            "mock_skill",
            {"runner": runner},
            {"query": "skill smoke check"},
        )
    )

    assert result["result"] == {"answer": "skill smoke check"}
    assert result["status"] == "completed"


def test_runtime_bridge_retries_transient_maf_failures() -> None:
    orchestrator = MockOrchestrator()
    attempts = 0

    class TransientGraph:
        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient graph failure")
            return {"summary": state["user_query"]}

    orchestrator.graph = TransientGraph()
    bridge = MAFSentinelRuntimeBridge(
        orchestrator,
        max_concurrent_skills=1,
        retry_policy=ExponentialRetryPolicy(
            max_attempts=1,
            base_delay=0,
            max_delay=0,
        ),
    )

    async def execute() -> dict[str, object]:
        await bridge.start()
        try:
            return await bridge.execute_workflow("Retry this workflow")
        finally:
            await bridge.stop()

    result = asyncio.run(execute())

    assert attempts == 2
    assert result["response"] == "Retry this workflow"
    assert result["status"] == "completed"


def test_runtime_bridge_reports_exhausted_maf_failures() -> None:
    orchestrator = MockOrchestrator()

    class FailingGraph:
        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            raise RuntimeError("workflow failed")

    orchestrator.graph = FailingGraph()
    bridge = MAFSentinelRuntimeBridge(
        orchestrator,
        max_concurrent_skills=1,
        retry_policy=ExponentialRetryPolicy(
            max_attempts=0,
            base_delay=0,
            max_delay=0,
        ),
    )

    async def execute() -> dict[str, object]:
        await bridge.start()
        try:
            return await bridge.execute_workflow("Fail this workflow")
        finally:
            await bridge.stop()

    result = asyncio.run(execute())

    assert result["status"] == "failed"
    assert result["error"] == "workflow failed"
    assert orchestrator.state_manager.get_scope("sentinel")["status"] == "failed"
