"""
Integration Tests - Basic tests for MAF + Sentinel integration

Usage:
    pytest tests/test_integration.py -v
    
    or
    
    python -m pytest tests/test_integration.py -v

DISABLED: this module imports `MAF.integration`, an external package that is not
part of this repo, plus `sentinal.core.execution_context` (moved to
`sentinal.core.execution.context`). Both fail at import, so collecting this file
aborted every pytest run. Commented out rather than deleted pending a decision on
whether to rewrite it against `sentinal/adapters/maf_adapter.py` with a fake
agent callable — MAFAdapter is duck-typed and needs no MAF package to test.
"""
#
# import pytest
# import asyncio
# from unittest.mock import Mock, AsyncMock, patch
#
# from MAF.integration import (
#     MAFWorkflowExecutable,
#     SkillExecutable,
#     SkillExecutor,
#     StateManagerBridge,
#     ExecuteSkillSignal,
# )
# from sentinal.core.execution_context import ExecutionContext
# from sentinal.core.execution.session import ExecutionSession
#
#
# class TestMAFWorkflowExecutable:
#     """Tests for MAFWorkflowExecutable."""
#
#     @pytest.mark.asyncio
#     async def test_execute_with_valid_context(self):
#         """Test workflow execution with valid context."""
#         # Mock orchestrator
#         orchestrator = Mock()
#         orchestrator.graph = AsyncMock(return_value={
#             "summary": "Test response",
#             "state": "final",
#         })
#
#         executable = MAFWorkflowExecutable(orchestrator)
#
#         context = ExecutionContext(
#             input={
#                 "user_query": "What is Python?",
#                 "memory_info": {"user_id": "test"},
#             }
#         )
#
#         result = await executable.execute(context)
#
#         assert result["status"] == "completed"
#         assert result["response"] == "Test response"
#         assert "execution_time_ms" in result
#         assert result["execution_id"] == str(context.execution_id)
#
#     @pytest.mark.asyncio
#     async def test_execute_with_error(self):
#         """Test workflow execution with error handling."""
#         orchestrator = Mock()
#         orchestrator.graph = AsyncMock(side_effect=ValueError("Test error"))
#
#         executable = MAFWorkflowExecutable(orchestrator)
#
#         context = ExecutionContext(input={"user_query": "test"})
#         result = await executable.execute(context)
#
#         assert result["status"] == "failed"
#         assert result["error"] == "Test error"
#         assert result["response"] is None
#
#
# class TestSkillExecutable:
#     """Tests for SkillExecutable."""
#
#     @pytest.mark.asyncio
#     async def test_execute_with_runner(self):
#         """Test skill execution with runner function."""
#         test_runner = AsyncMock(return_value={"output": "test"})
#
#         executable = SkillExecutable(
#             skill_name="test_skill",
#             runner=test_runner,
#         )
#
#         context = ExecutionContext(
#             input={"query": "test"}
#         )
#
#         result = await executable.execute(context)
#
#         assert result["status"] == "completed"
#         assert result["executed"] is True
#         assert result["skill_name"] == "test_skill"
#         assert result["result"]["output"] == "test"
#
#     @pytest.mark.asyncio
#     async def test_execute_with_graph(self):
#         """Test skill execution with LangGraph graph."""
#         test_graph = Mock()
#         test_graph.ainvoke = AsyncMock(return_value={"output": "graph_result"})
#
#         executable = SkillExecutable(
#             skill_name="graph_skill",
#             graph=test_graph,
#         )
#
#         context = ExecutionContext(input={"query": "test"})
#         result = await executable.execute(context)
#
#         assert result["status"] == "completed"
#         assert result["result"]["output"] == "graph_result"
#
#     @pytest.mark.asyncio
#     async def test_execute_with_error(self):
#         """Test skill execution with error."""
#         test_runner = AsyncMock(side_effect=RuntimeError("Skill failed"))
#
#         executable = SkillExecutable(
#             skill_name="failing_skill",
#             runner=test_runner,
#         )
#
#         context = ExecutionContext(input={})
#         result = await executable.execute(context)
#
#         assert result["status"] == "failed"
#         assert result["executed"] is False
#         assert result["error"] == "Skill failed"
#
#     @pytest.mark.asyncio
#     async def test_state_manager_integration(self):
#         """Test skill executable stores result in state manager."""
#         state_manager = Mock()
#         state_manager.update = Mock()
#
#         test_runner = AsyncMock(return_value={"output": "result"})
#
#         executable = SkillExecutable(
#             skill_name="tracked_skill",
#             runner=test_runner,
#             state_manager=state_manager,
#             scope="test_scope",
#         )
#
#         context = ExecutionContext(input={})
#         await executable.execute(context)
#
#         # Verify state manager was called
#         state_manager.update.assert_called_once()
#         call_args = state_manager.update.call_args
#         assert call_args[0][0] == "test_scope"
#
#
# class TestStateManagerBridge:
#     """Tests for StateManagerBridge."""
#
#     def test_get_combined_state(self):
#         """Test getting combined state view."""
#         state_manager = Mock()
#         state_manager.global_state = {"user": "test_user"}
#         state_manager.scopes = {
#             "agent": {"intent": "search"},
#             "skill": {"result": "data"},
#         }
#
#         session = Mock()
#         session.execution_id = "test-id"
#         session.session_id = "session-id"
#         session.state = Mock()
#         session.state.status = Mock(value="RUNNING")
#
#         bridge = StateManagerBridge(state_manager, session)
#         combined = bridge.get_combined_state()
#
#         assert "maf" in combined
#         assert "sentinel" in combined
#         assert combined["maf"]["global"]["user"] == "test_user"
#         assert combined["sentinel"]["execution_id"] == "test-id"
#
#     def test_store_skill_result(self):
#         """Test storing skill result in bridge."""
#         state_manager = Mock()
#         state_manager.update = Mock()
#
#         session = Mock()
#         session.execution_id = "test-id"
#         session.runtime_data = {}
#
#         bridge = StateManagerBridge(state_manager, session)
#
#         bridge.store_skill_result(
#             skill_name="web_search",
#             result={
#                 "result": {"output": "data"},
#                 "execution_time_ms": 123.4,
#                 "status": "completed",
#                 "executed": True,
#             },
#         )
#
#         state_manager.update.assert_called_once()
#
#
# class TestSkillExecutor:
#     """Tests for SkillExecutor actor."""
#
#     @pytest.mark.asyncio
#     async def test_skill_executor_initialization(self):
#         """Test SkillExecutor can be created."""
#         executor = SkillExecutor()
#         assert executor is not None
#
#
# class TestExecuteSkillSignal:
#     """Tests for ExecuteSkillSignal."""
#
#     def test_signal_creation(self):
#         """Test creating an ExecuteSkillSignal."""
#         signal = ExecuteSkillSignal(
#             skill_name="test_skill",
#             skill_entry={"runner": Mock()},
#             state={"query": "test"},
#             execution_id="test-id",
#         )
#
#         assert signal.skill_name == "test_skill"
#         assert signal.execution_id == "test-id"
#
#
# # Integration-level tests
# class TestIntegration:
#     """Integration tests combining multiple components."""
#
#     @pytest.mark.asyncio
#     async def test_workflow_execution_flow(self):
#         """Test complete workflow execution."""
#         # Mock orchestrator
#         orchestrator = Mock()
#         orchestrator.graph = AsyncMock(return_value={
#             "summary": "Workflow complete",
#         })
#         orchestrator.state_manager = Mock()
#
#         # Create executable
#         executable = MAFWorkflowExecutable(orchestrator)
#
#         # Create and execute context
#         context = ExecutionContext(
#             input={
#                 "user_query": "Integration test",
#                 "memory_info": {},
#             }
#         )
#
#         result = await executable.execute(context)
#
#         assert result["status"] == "completed"
#         assert result["response"] == "Workflow complete"
#         assert orchestrator.graph.called
#
#     @pytest.mark.asyncio
#     async def test_skill_execution_flow(self):
#         """Test complete skill execution."""
#         state_manager = Mock()
#         state_manager.update = Mock()
#
#         runner = AsyncMock(return_value={"data": "result"})
#
#         executable = SkillExecutable(
#             skill_name="integration_skill",
#             runner=runner,
#             state_manager=state_manager,
#         )
#
#         context = ExecutionContext(input={"query": "test"})
#         result = await executable.execute(context)
#
#         assert result["executed"] is True
#         assert result["skill_name"] == "integration_skill"
#         state_manager.update.assert_called_once()
#
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
#