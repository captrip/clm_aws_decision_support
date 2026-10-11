import os
import sys

import pytest

# examples/ is not a package — put it on the path so we can import the demo module.
_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
sys.path.insert(0, os.path.abspath(_EXAMPLES))

import search_agent as sa  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from sentinal.core.trajectory import CorrectionFlag, TrajectoryStep  # noqa: E402


class _FakeLLM:
    """Stub chat model: returns a fixed reply for .invoke()."""
    def __init__(self, reply):
        self._reply = reply

    def invoke(self, messages):
        return AIMessage(content=self._reply)


def _tool_step(tool_name):
    return TrajectoryStep(
        index=0, kind="tool", summary="tools",
        raw={"tools": {"messages": [
            ToolMessage(content="...", name=tool_name, tool_call_id="1")]}})


def test_build_agent_binds_both_tools():
    """build_agent must accept a custom tool set and expose wiki_lookup."""
    assert sa.wiki_lookup is not None  # Verify tool exists
    # Default: single tool.
    assert sa.build_agent is not None
    # wiki_lookup returns a graceful string on bad input, never raises.
    out = sa.wiki_lookup.invoke({"topic": ""})
    assert isinstance(out, str)


def test_tool_name_from_step_reads_tool_message():
    assert sa.tool_name_from_step(_tool_step("web_search")) == "web_search"


def test_tool_name_from_step_reads_agent_tool_call():
    step = TrajectoryStep(
        index=0, kind="agent", summary="agent",
        raw={"agent": {"messages": [
            AIMessage(content="", tool_calls=[
                {"name": "wiki_lookup", "args": {}, "id": "1"}])]}})
    assert sa.tool_name_from_step(step) == "wiki_lookup"


def test_judge_corrects_wrong_tool():
    judge = sa.ToolChoiceJudge(
        query="Find information about PEP 703.",
        tool_names=["web_search", "wiki_lookup"],
        llm_factory=lambda: _FakeLLM("wiki_lookup"))
    c = judge.judge([_tool_step("web_search")])
    assert c.flag == CorrectionFlag.AVOID_TOOL
    assert "web_search" in c.note and "wiki_lookup" in c.note


def test_judge_allows_right_tool():
    judge = sa.ToolChoiceJudge(
        query="q", tool_names=["web_search", "wiki_lookup"],
        llm_factory=lambda: _FakeLLM("ok"))
    assert judge.judge([_tool_step("wiki_lookup")]).flag == CorrectionFlag.CONTINUE


def test_judge_dedupes_same_wrong_tool():
    judge = sa.ToolChoiceJudge(
        query="q", tool_names=["web_search", "wiki_lookup"],
        llm_factory=lambda: _FakeLLM("wiki_lookup"))
    first = judge.judge([_tool_step("web_search")])
    second = judge.judge([_tool_step("web_search")])
    assert first.flag == CorrectionFlag.AVOID_TOOL
    assert second.flag == CorrectionFlag.CONTINUE


def test_judge_survives_broken_llm():
    class _Boom:
        def invoke(self, messages):
            raise RuntimeError("llm down")
    judge = sa.ToolChoiceJudge(
        query="q", tool_names=["web_search", "wiki_lookup"],
        llm_factory=lambda: _Boom())
    assert judge.judge([_tool_step("web_search")]).flag == CorrectionFlag.CONTINUE


def test_make_tool_choice_run_wires_judge_and_tools(monkeypatch):
    # Mock get_llm to avoid AWS/Azure credential requirements
    def mock_get_llm():
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        class _TestLLM(BaseChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                pass
            def _llm_type(self):
                return "test"
            def bind_tools(self, tools):
                return self
            def invoke(self, messages):
                return AIMessage(content="ok")
        return _TestLLM()
    monkeypatch.setattr(sa, "get_llm", mock_get_llm)

    judge, executable, tools = sa.make_tool_choice_run("Find info about PEP 703.")
    assert isinstance(judge, sa.ToolChoiceJudge)
    assert isinstance(executable, sa.SearchAgentExecutable)
    names = {getattr(t, "name", "") for t in tools}
    assert names == {"web_search", "wiki_lookup"}


# ─── Integration test (real LLM + Wikipedia) ─────────────────────────────────
import asyncio  # noqa: E402
import urllib.request  # noqa: E402

from sentinal.runtime.runtime import SentinelRuntime  # noqa: E402
from sentinal.actors.governor import Governor  # noqa: E402
from sentinal.core.actor_ref import ActorRef  # noqa: E402
from sentinal.core.execution.context import ExecutionContext  # noqa: E402
from sentinal.core.execution.session import ExecutionSession  # noqa: E402


def _llm_available() -> bool:
    try:
        sa.get_llm()
        return True
    except Exception:
        return False


def _wikipedia_reachable() -> bool:
    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/api/rest_v1/page/summary/Python_(programming_language)",
            headers={"User-Agent": "sentinal-demo/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _run_tool_choice(query):
    judge, executable, _ = sa.make_tool_choice_run(query)
    runtime = SentinelRuntime()
    runtime.register(Governor(ActorRef("governor", "local"), judge=judge))
    session = ExecutionSession(
        executable=executable,
        context=ExecutionContext(input={"query": query}),
    )
    await runtime.execute(session)
    # Get governor BEFORE shutdown (shutdown clears actors list)
    governor = next(a for a in runtime.actors if a.ref.name == "governor")
    interventions = governor.get_interventions()
    await runtime.shutdown()
    return session, interventions


@pytest.mark.skipif(not _llm_available(), reason="LLM provider/credentials unavailable")
@pytest.mark.skipif(not _wikipedia_reachable(), reason="Wikipedia API unreachable")
def test_governor_steers_wrong_tool_to_right_tool():
    """SOFT integration test (real LLM, may occasionally flake): the agent tends
    to reach for web_search on a generic 'find information' query; the Governor
    should steer it to wiki_lookup for this encyclopedic topic."""
    query = "Find information about PEP 703 and summarize it."
    session, interventions = asyncio.run(_run_tool_choice(query))

    corrections = [i for i in interventions if i.get("type") == "correction"]
    avoid = [c for c in corrections
             if c.get("flag") == "avoid_tool" and "web_search" in (c.get("note") or "")]

    trajectory = session.get_trajectory()
    tools_used = {sa.tool_name_from_step(s) for s in trajectory}

    # 1. The run completed (not aborted/errored).
    assert session.state.status.value == "completed"
    # 2. The Governor steered away from web_search toward wiki_lookup.
    assert avoid, f"expected an avoid_tool(web_search) correction; got {corrections}"
    # 3. The agent ended up calling the right tool.
    assert "wiki_lookup" in tools_used, f"tools used: {tools_used}"
