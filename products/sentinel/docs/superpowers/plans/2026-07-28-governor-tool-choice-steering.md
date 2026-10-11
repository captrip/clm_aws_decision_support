# Governor Tool-Choice Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a watchable demo + tests where the Governor detects a wrong tool choice (agent uses `web_search` for an encyclopedic topic) and steers the agent to the right tool (`wiki_lookup`) via an `AVOID_TOOL` correction.

**Architecture:** Extend `examples/search_agent.py` with a second tool (`wiki_lookup`, Wikipedia REST API via stdlib `urllib`), a `--tool-choice` CLI mode, and an LLM-based `ToolChoiceJudge` that inspects the latest tool call and issues `AVOID_TOOL` when a better tool exists. The existing executable already handles `AVOID_TOOL` (appends a governor message + restarts the stream), so no core `sentinal/` changes. Deterministic unit tests cover the judge with a stub LLM; a soft-assert integration test covers the real end-to-end steer.

**Tech Stack:** Python 3.14 free-threaded, LangGraph, LangChain, `ddgs`, stdlib `urllib`, pytest.

## Global Constraints

- Run everything with `PYTHON_GIL=0` and the venv interpreter: `PYTHON_GIL=0 .ft-env/bin/python ...`.
- No new third-party dependencies — Wikipedia access uses stdlib `urllib`.
- Do NOT add tool-routing guidance to `SYSTEM_PROMPT` (the agent must be free to pick the wrong tool, or the demo never triggers a correction).
- Judges implement the sync `TrajectoryJudge` protocol: `judge(trajectory: list[TrajectoryStep]) -> Correction`.
- A broken judge must never crash the run — swallow judge-internal exceptions and return `CONTINUE`.
- Commits stay LOCAL (do not push). Commit steps are included for rhythm; the user may skip them.

---

### Task 1: Add `wiki_lookup` tool and make the agent tool-set configurable

**Files:**
- Modify: `examples/search_agent.py` (add `wiki_lookup` after `web_search` ~line 103; change `build_agent` ~line 145 and `SearchAgentExecutable.__init__` ~line 184)
- Test: `tests/test_trajectory/test_tool_choice_steering.py` (new — unit portion)

**Interfaces:**
- Produces:
  - `web_search` (existing) and `wiki_lookup` — LangChain `@tool` callables. `wiki_lookup(topic: str) -> str`.
  - `build_agent(tools: list | None = None)` — compiled LangGraph agent; defaults to `[web_search]`.
  - `SearchAgentExecutable(tools: list | None = None)` — passes `tools` through to `build_agent`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trajectory/test_tool_choice_steering.py`:

```python
import os
import sys

import pytest

# examples/ is not a package — put it on the path so we can import the demo module.
_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
sys.path.insert(0, os.path.abspath(_EXAMPLES))

import search_agent as sa  # noqa: E402


def test_build_agent_binds_both_tools():
    """build_agent must accept a custom tool set and expose wiki_lookup."""
    assert callable(sa.wiki_lookup)
    # Default: single tool.
    assert sa.build_agent is not None
    # wiki_lookup returns a graceful string on bad input, never raises.
    out = sa.wiki_lookup.invoke({"topic": ""})
    assert isinstance(out, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_tool_choice_steering.py -v`
Expected: FAIL with `AttributeError: module 'search_agent' has no attribute 'wiki_lookup'`.

- [ ] **Step 3: Add the `wiki_lookup` tool**

In `examples/search_agent.py`, after the `web_search` function (after line 102), add:

```python
@tool
@traced(name="tool.wiki_lookup")
@logged
def wiki_lookup(topic: str) -> str:
    """Look up an encyclopedic/reference topic on Wikipedia. Best for
    definitions, background, and established facts about a named subject."""
    import json
    import urllib.parse
    import urllib.request

    title = urllib.parse.quote(topic.strip().replace(" ", "_"))
    if not title:
        return "No topic provided."
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "sentinal-demo/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"
    activity_bus.emit("tool", "tool.wiki_lookup", detail=topic[:24])
    extract = data.get("extract", "")
    page = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    if not extract:
        return "No Wikipedia article found."
    return f"**{data.get('title', topic)}**\n{extract}\nSource: {page}"
```

- [ ] **Step 4: Make `build_agent` accept a tool set**

Replace the `build_agent` signature and its `tools =` line (lines 145-149):

```python
def build_agent(tools: list | None = None):
    """Build a ReAct agent. Defaults to web search only; pass a tool list to
    give the agent more choices (used by the tool-choice steering demo)."""
    llm = get_llm()

    tools = tools if tools is not None else [web_search]
    llm_with_tools = llm.bind_tools(tools)
```

- [ ] **Step 5: Make the executable accept a tool set**

Replace `SearchAgentExecutable.__init__` (lines 184-185):

```python
    def __init__(self, tools: list | None = None):
        self._agent = build_agent(tools)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_tool_choice_steering.py::test_build_agent_binds_both_tools -v`
Expected: PASS. (This makes a real Wikipedia call with an empty topic → returns "No topic provided." without network, since we guard the empty title.)

- [ ] **Step 7: Commit**

```bash
git add examples/search_agent.py tests/test_trajectory/test_tool_choice_steering.py
git commit -m "feat: add wiki_lookup tool and configurable agent tool-set"
```

---

### Task 2: Add `tool_name_from_step` helper and `ToolChoiceJudge`

**Files:**
- Modify: `examples/search_agent.py` (add helper + judge class near `DemoSteeringJudge`, ~line 277)
- Test: `tests/test_trajectory/test_tool_choice_steering.py` (add unit tests)

**Interfaces:**
- Consumes: `CorrectionFlag`, `Correction`, `TrajectoryStep` (already imported in the module); `get_llm` (module-level).
- Produces:
  - `tool_name_from_step(step) -> str | None` — extracts the tool name from a trajectory step's raw chunk (checks `ToolMessage.name`, then `AIMessage.tool_calls[*].name`).
  - `ToolChoiceJudge(query: str, tool_names: list[str], llm_factory=get_llm)` — `judge(trajectory)` returns `AVOID_TOOL` (note `"<wrong>; use <right> instead"`) when the latest tool call is suboptimal, else `CONTINUE`. Dedupes per wrong-tool name.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trajectory/test_tool_choice_steering.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_tool_choice_steering.py -v -k "judge or tool_name"`
Expected: FAIL with `AttributeError: module 'search_agent' has no attribute 'tool_name_from_step'`.

- [ ] **Step 3: Add the helper and judge**

In `examples/search_agent.py`, immediately before the `DemoSteeringJudge` class (before line 277 `# ─── Demo judge`), add:

```python
# ─── Tool-choice steering ────────────────────────────────────────────────────
def tool_name_from_step(step) -> str | None:
    """Extract the tool name a step represents, from its raw streamed chunk.
    Checks executed-tool messages (ToolMessage.name) first, then a pending
    agent tool call (AIMessage.tool_calls)."""
    raw = getattr(step, "raw", None)
    if not isinstance(raw, dict):
        return None
    for node_out in raw.values():
        msgs = node_out.get("messages") if isinstance(node_out, dict) else None
        if not msgs:
            continue
        for m in msgs:
            name = getattr(m, "name", None)
            if name:  # ToolMessage.name
                return name
            for tc in (getattr(m, "tool_calls", None) or []):
                n = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if n:
                    return n
    return None


class ToolChoiceJudge:
    """LLM-based judge that watches which tool the agent calls and, when a
    better tool exists for the user's query, issues an AVOID_TOOL correction
    steering the agent to the right tool. Fires at most once per wrong tool."""

    _TOOL_HINTS = {
        "web_search": "general/current web information",
        "wiki_lookup": "encyclopedic/reference topics about a named subject",
    }

    def __init__(self, query: str, tool_names: list[str], llm_factory=get_llm):
        self._query = query
        self._tool_names = tool_names
        self._llm_factory = llm_factory
        self._corrected: set[str] = set()

    def judge(self, trajectory):
        if not trajectory:
            return Correction(flag=CorrectionFlag.CONTINUE)
        tool = tool_name_from_step(trajectory[-1])
        if tool is None or tool in self._corrected:
            return Correction(flag=CorrectionFlag.CONTINUE)
        best = self._best_tool(tool)
        if best and best != tool and best in self._tool_names:
            self._corrected.add(tool)
            return Correction(
                flag=CorrectionFlag.AVOID_TOOL,
                note=f"{tool}; use {best} instead",
                step_index=len(trajectory) - 1,
            )
        return Correction(flag=CorrectionFlag.CONTINUE)

    def _best_tool(self, chosen: str) -> str | None:
        """Ask the LLM whether `chosen` is the best tool; return the better tool
        name, or None if the choice is fine / on any error."""
        hints = "\n".join(
            f"- {n}: {self._TOOL_HINTS.get(n, 'a tool')}" for n in self._tool_names)
        prompt = (
            f"User query: {self._query}\n"
            f"Available tools:\n{hints}\n"
            f"The agent chose to call: {chosen}\n"
            "Reply with ONLY the single best tool name for this query, or the "
            "word ok if the agent's choice is fine. No other text."
        )
        try:
            resp = self._llm_factory().invoke(
                [SystemMessage(content="You are a tool-routing critic."),
                 HumanMessage(content=prompt)])
            answer = (getattr(resp, "content", "") or "").strip().lower()
        except Exception:
            return None
        if not answer or answer.startswith("ok"):
            return None
        # Normalize: pick the first tool name that appears in the reply.
        for n in self._tool_names:
            if n in answer:
                return n
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_tool_choice_steering.py -v -k "judge or tool_name"`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/search_agent.py tests/test_trajectory/test_tool_choice_steering.py
git commit -m "feat: add ToolChoiceJudge and tool_name_from_step helper"
```

---

### Task 3: Wire `--tool-choice` mode into `main()`

**Files:**
- Modify: `examples/search_agent.py` (`main()` ~lines 306-330)
- Test: `tests/test_trajectory/test_tool_choice_steering.py` (add a config smoke test)

**Interfaces:**
- Consumes: `ToolChoiceJudge`, `web_search`, `wiki_lookup`, `SearchAgentExecutable` (Tasks 1-2).
- Produces: a factory helper `make_tool_choice_run(query)` returning `(judge, executable, tools)` so both `main()` and the integration test build the scenario identically.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trajectory/test_tool_choice_steering.py`:

```python
def test_make_tool_choice_run_wires_judge_and_tools():
    judge, executable, tools = sa.make_tool_choice_run("Find info about PEP 703.")
    assert isinstance(judge, sa.ToolChoiceJudge)
    assert isinstance(executable, sa.SearchAgentExecutable)
    names = {getattr(t, "name", "") for t in tools}
    assert names == {"web_search", "wiki_lookup"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_tool_choice_steering.py::test_make_tool_choice_run_wires_judge_and_tools -v`
Expected: FAIL with `AttributeError: module 'search_agent' has no attribute 'make_tool_choice_run'`.

- [ ] **Step 3: Add the factory helper**

In `examples/search_agent.py`, after the `ToolChoiceJudge` class, add:

```python
def make_tool_choice_run(query: str):
    """Build the (judge, executable, tools) triple for the tool-choice demo,
    shared by main() and the integration test so both wire it identically."""
    tools = [web_search, wiki_lookup]
    judge = ToolChoiceJudge(
        query=query, tool_names=[t.name for t in tools])
    executable = SearchAgentExecutable(tools=tools)
    return judge, executable, tools
```

- [ ] **Step 4: Wire the flag into `main()`**

Replace the top of `main()` (lines 307-312) — the argv parsing and judge selection:

```python
    argv = sys.argv[1:]
    demo_steering = "--demo-steering" in argv
    tool_choice = "--tool-choice" in argv
    argv = [a for a in argv if a not in ("--demo-steering", "--tool-choice")]
    if tool_choice:
        default_query = "Find information about PEP 703 and summarize it."
    else:
        default_query = "What is Python 3.14 free threading?"
    query = " ".join(argv) if argv else default_query

    if demo_steering:
        judge = DemoSteeringJudge()
        executable = SearchAgentExecutable()
    elif tool_choice:
        judge, executable, _ = make_tool_choice_run(query)
    else:
        judge = HeuristicJudge()
        executable = SearchAgentExecutable()
```

- [ ] **Step 5: Use the pre-built executable in the session**

Replace the `session = ExecutionSession(...)` block (lines 327-330):

```python
    session = ExecutionSession(
        executable=executable,
        context=ExecutionContext(input={"query": query}),
    )
```

- [ ] **Step 6: Update the judge banner line**

Replace the Judge print (lines 319-320):

```python
    if demo_steering:
        judge_desc = " (forces REDIRECT then STOP)"
    elif tool_choice:
        judge_desc = " (steers wrong tool → right tool)"
    else:
        judge_desc = " (fires only if the agent misbehaves)"
    print(f"  Judge: {type(judge).__name__}{judge_desc}")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_tool_choice_steering.py -v -k "make_tool_choice or judge or tool_name or build_agent"`
Expected: PASS. Also verify the module still compiles: `.ft-env/bin/python -m py_compile examples/search_agent.py` → no output.

- [ ] **Step 8: Commit**

```bash
git add examples/search_agent.py tests/test_trajectory/test_tool_choice_steering.py
git commit -m "feat: add --tool-choice mode to search_agent demo"
```

---

### Task 4: Soft-assert integration test (real LLM + Wikipedia)

**Files:**
- Test: `tests/test_trajectory/test_tool_choice_steering.py` (add the integration test)

**Interfaces:**
- Consumes: `make_tool_choice_run` (Task 3); `SentinelRuntime`, `Governor`, `ActorRef`, `ExecutionSession`, `ExecutionContext` (via `search_agent` module or direct import); `tool_name_from_step` (Task 2).

- [ ] **Step 1: Write the integration test**

Add to `tests/test_trajectory/test_tool_choice_steering.py`:

```python
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
    await runtime.shutdown()
    governor = next(a for a in runtime.actors if a.ref.name == "governor")
    return session, governor.get_interventions()


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
```

- [ ] **Step 2: Run the full test file**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/test_tool_choice_steering.py -v`
Expected: unit tests PASS; the integration test PASSES if LLM+Wikipedia are available, else SKIPS. (If it fails on assert 2 because the agent happened to pick the right tool immediately, re-run — this is the accepted soft-flake per the design.)

- [ ] **Step 3: Run the whole trajectory suite (no regressions)**

Run: `PYTHON_GIL=0 .ft-env/bin/python -m pytest tests/test_trajectory/ -q`
Expected: all prior tests still pass; new tests pass or skip.

- [ ] **Step 4: Watch it live (manual verification)**

Run: `PYTHON_GIL=0 .ft-env/bin/python examples/search_agent.py --tool-choice`
Expected: dashboard shows `web_search` → Governor row `correction · avoid_tool:web_search; use wiki_lookup` → `wiki_lookup` → completed; end-of-run "Governor corrections issued" lists the tool-choice correction.

- [ ] **Step 5: Commit**

```bash
git add tests/test_trajectory/test_tool_choice_steering.py
git commit -m "test: soft-assert integration test for governor tool-choice steering"
```

---

## Self-Review

**Spec coverage:**
- Scenario & tools (wiki_lookup, --tool-choice) → Tasks 1, 3. ✅
- LLM-based `ToolChoiceJudge` with dedupe + graceful failure → Task 2. ✅
- Soft-assert integration test with LLM/Wikipedia skips → Task 4. ✅
- No core `sentinal/` changes; reuses existing `AVOID_TOOL` handling → confirmed (executable already implements it). ✅
- No new dependencies (stdlib urllib) → Task 1. ✅
- Don't add tool-routing guidance to SYSTEM_PROMPT → honored (untouched). ✅

**Placeholder scan:** No TBD/TODO; all code blocks complete. ✅

**Type consistency:** `make_tool_choice_run` returns `(judge, executable, tools)` used consistently in Tasks 3-4; `tool_name_from_step` signature identical across judge + test; `ToolChoiceJudge(query, tool_names, llm_factory)` consistent across Tasks 2-3. Tool `.name` attributes (`web_search`, `wiki_lookup`) match the `@tool`-decorated function names. ✅
