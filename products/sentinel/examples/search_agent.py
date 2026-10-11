"""
Live LangGraph Agent with Web Search — running through Sentinal Runtime

Demonstrates:
1. A LangGraph ReAct agent with DuckDuckGo search tool
2. Wrapped as a Sentinal Executable with @traced and @logged decorators
3. Dispatched via HybridRunner on real OS threads (GIL disabled)
4. Full observability: structured logs + OTel-compatible tracing
5. Pluggable judges: the Governor judges the streaming trajectory with a LIST of
   judges — the curated built-ins alongside business judges defined right here in
   this file via @judge — and pushes the most severe correction to the executable
   between steps (visible in the live dashboard's Governor row). Every judge's
   finding is recorded, so you can see which won arbitration and which lost.

Usage:
    # PYTHONPATH=. is REQUIRED: an installed `sentinal` in site-packages would
    # otherwise shadow this repo, and the new judge API would not import.

    # AWS Bedrock (default):
    export LLM_PROVIDER=aws
    export AWS_PROFILE="258574424891"

    # Azure OpenAI:
    export LLM_PROVIDER=azure
    export AZURE_OPENAI_API_KEY="your-key"
    export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
    export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
    export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

    # Normal run — production budgets. The four built-in judges plus two
    # user-defined business judges stay silent unless the agent misbehaves
    # (repeats a tool, blows the step or latency budget, loops, answers at
    # length, or overspends its search budget):
    PYTHON_GIL=0 PYTHONPATH=. .ft-env/bin/python examples/search_agent.py \
        "What is Python 3.14 free threading?"

    # ► START HERE to see the feature work. Same judge roster, budgets shrunk so
    # a WELL-BEHAVED agent trips them. Prints the roster (built-in vs custom),
    # then every finding: which judge raised it, and whether it was pushed to the
    # agent or lost severity arbitration to a more severe one:
    PYTHON_GIL=0 PYTHONPATH=. .ft-env/bin/python examples/search_agent.py --tight-budgets

    # Deterministic steering — a DemoSteeringJudge issues REDIRECT then STOP, so
    # you can watch the Governor intervene and then abort the run:
    PYTHON_GIL=0 PYTHONPATH=. .ft-env/bin/python examples/search_agent.py \
        --demo-steering "Tell me about asyncio"

    # An LLM-backed judge that redirects a wrong tool pick to the right tool:
    PYTHON_GIL=0 PYTHONPATH=. .ft-env/bin/python examples/search_agent.py --tool-choice

    # Add SENTINAL_LIVE_DASHBOARD=1 to any of the above for the live 3-thread view.
"""

import asyncio
import logging
import os
import re
import sys
import threading
import time

os.environ.setdefault("AWS_PROFILE", "258574424891")

from dotenv import load_dotenv
load_dotenv()

from contextlib import aclosing

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from ddgs import DDGS
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

# Sentinal imports
from sentinal import (
    DevicePreference,
    HybridRunner,
    configure,
    traced,
    logged,
    LiveThreadDashboard,
    activity_bus,
    Correction,
    CorrectionFlag,
    TrajectoryStep,
    LatencyBudgetJudge,
    OutputLoopJudge,
    RepeatToolJudge,
    StepBudgetJudge,
    finalize,
    judge,
    redirect,
)
from sentinal.telemetry import dashboard_enabled
from sentinal.core.executable import Executable
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.session import ExecutionSession
from sentinal.runtime.runtime import SentinelRuntime
from sentinal.exceptions.runtime import ExecutionAborted
# ─── Configure observability ─────────────────────────────────────────────────
configure(
    service_name="search-agent",
    log_level="INFO",
    log_format="text",
    trace_exporter="none",  # set to "console" to see spans
    live_dashboard=True,
)

logger = logging.getLogger("search_agent")


# ─── Define the search tool ──────────────────────────────────────────────────
@tool
@traced(name="tool.web_search")
@logged
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Use this to find current information."""
    results = DDGS().text(query, max_results=3)
    if not results:
        return "No results found."
    activity_bus.emit("tool", "tool.web_search", detail=query[:24])
    output = []
    for r in results:
        output.append(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}\n")
    return "\n".join(output)


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


# ─── LLM provider factory ───────────────────────────────────────────────────
def get_llm():
    """Return a LangChain chat model based on LLM_PROVIDER env var."""
    provider = os.environ.get("LLM_PROVIDER", "aws").lower()

    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=0,
        )
    elif provider == "aws":
        try:
            from aws_bedrock_token_generator import provide_token
        except ImportError as error:
            raise RuntimeError(
                "AWS mode requires aws_bedrock_token_generator; install the "
                "provider integration before selecting LLM_PROVIDER=aws."
            ) from error
        return ChatOpenAI(
            model=os.environ.get("AWS_BEDROCK_MODEL", "openai.gpt-oss-120b"),
            api_key=provide_token(region=os.environ.get("AWS_REGION", "us-east-1")),
            base_url="https://bedrock-mantle.{}.api.aws/v1".format(
                os.environ.get("AWS_REGION", "us-east-1")
            ),
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'aws' or 'azure'.")


# ─── System prompt ───────────────────────────────────────────────────────────
# The agent had no system prompt, so it answered from parametric memory and
# invented specifics (fake CLI flags, C-API names, URLs). Ground it in the
# web_search results and forbid fabrication.
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a research assistant. Answer ONLY from the web_search results in "
    "this conversation. Do not rely on prior knowledge for specific facts such "
    "as flag names, API/function names, version numbers, dates, or URLs — if a "
    "detail is not in the search results, either search for it or say you could "
    "not confirm it. Never invent command-line flags, environment variables, "
    "C-API symbols, or links. Cite the source URL for concrete claims, and keep "
    "the answer concise."
))


# ─── Build the LangGraph agent ───────────────────────────────────────────────
def build_agent(tools: list | None = None):
    """Build a ReAct agent. Defaults to web search only; pass a tool list to
    give the agent more choices (used by the tool-choice steering demo)."""
    llm = get_llm()

    tools = tools if tools is not None else [web_search]
    llm_with_tools = llm.bind_tools(tools)
    def call_model(state: MessagesState):
        # Prepend the grounding system prompt (once) so every model turn stays
        # anchored to the search results.
        messages = state["messages"]
        if not (messages and isinstance(messages[0], SystemMessage)):
            messages = [SYSTEM_PROMPT, *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()


# ─── Sentinal Executable wrapper ─────────────────────────────────────────────
class SearchAgentExecutable(Executable):
    """Wraps the LangGraph agent as a Sentinal Executable."""

    device = DevicePreference.CPU  # LLM calls are IO-bound, run on CPU thread

    def __init__(self, tools: list | None = None):
        self._agent = build_agent(tools)

    @traced(name="agent.execute")
    @logged(capture_result=False)
    async def execute(self, context):
        query = context.input.get("query", "Hello")
        session = context.session
        # The running conversation. Corrections that steer (REDIRECT / AVOID_TOOL)
        # append a governor message and restart the stream so the agent actually
        # sees the nudge; FORCE_FINALIZE stops streaming and returns what we have.
        messages = [HumanMessage(content=query)]
        i = 0
        finalized = False
        # Outer loop: each pass runs one astream() to (early) completion. We break
        # out of the inner stream on a steering/finalize correction because a live
        # astream() generator captured its inputs at creation — reassigning a local
        # would be a no-op, so we must re-enter astream() with updated messages.
        while True:
            restart = False
            # aclosing() guarantees the generator is closed when we break early,
            # so LangGraph's stream task doesn't leak ("Task was destroyed but it
            # is pending" / "aclose was never awaited").
            async with aclosing(self._agent.astream(
                    {"messages": messages}, stream_mode="updates")) as stream:
                async for chunk in stream:
                    # Accumulate produced messages so a restart keeps full ReAct context.
                    for v in chunk.values():
                        if isinstance(v, dict) and v.get("messages"):
                            messages.extend(v["messages"])
                    if session is not None:
                        node = next(iter(chunk)) if chunk else "agent"
                        kind = "tool" if node == "tools" else "agent"
                        session.append_step(TrajectoryStep(
                            index=i, kind=kind, summary=str(node),
                            raw=chunk, ts=time.monotonic()))
                        for c in session.drain_corrections():
                            if c.flag == CorrectionFlag.STOP:
                                raise ExecutionAborted(c.note or "stopped by governor")
                            elif c.flag == CorrectionFlag.FORCE_FINALIZE:
                                finalized = True
                            elif c.flag in (CorrectionFlag.REDIRECT,
                                            CorrectionFlag.AVOID_TOOL):
                                messages.append(HumanMessage(
                                    content=f"[governor:{c.flag.value}] {c.note or 'adjust course'}"))
                                restart = True
                        if finalized or restart:
                            break
                    i += 1
            if finalized or not restart:
                break
        # If the Governor forced finalize mid-ReAct, the newest messages may be
        # raw tool output (a ToolMessage) or an AIMessage that only holds a
        # pending tool_call — never present those as the answer. Ask the agent to
        # wrap up now, without tools, so the user gets a synthesized reply.
        if finalized:
            messages.append(HumanMessage(
                content="[governor:force_finalize] Stop searching and answer now "
                        "using what you already have."))
            final_text = await self._finalize(messages)
        else:
            final_text = self._latest_answer(messages)
        return {"query": query, "response": final_text,
                "steps": i, "thread_id": threading.current_thread().native_id}

    async def _finalize(self, messages: list) -> str:
        """One tool-free LLM pass to synthesize an answer from accumulated context."""
        try:
            grounded = self._drop_dangling_tool_calls(messages)
            if not (grounded and isinstance(grounded[0], SystemMessage)):
                grounded = [SYSTEM_PROMPT, *grounded]
            resp = await get_llm().ainvoke(grounded)
            text = getattr(resp, "content", "") or ""
            if text:
                return text
        except Exception as e:
            logger.warning("finalize pass failed, using accumulated text: %s", e)
        return self._latest_answer(messages) or "(force-finalized before a synthesized answer)"

    @staticmethod
    def _drop_dangling_tool_calls(messages: list) -> list:
        """Remove tool-call turns whose results never arrived.

        A correction can break the stream between the model requesting a tool and
        the ToolNode answering it. Chat APIs reject that: "an assistant message
        with 'tool_calls' must be followed by tool messages responding to each
        tool_call_id" — a 400 that would sink the whole finalize pass. Drop the
        unanswered request (and any orphaned result) instead; the retrieved
        content the agent already summarized stays in the transcript.
        """
        answered = {
            tc_id for m in messages
            if isinstance(m, ToolMessage)
            and (tc_id := getattr(m, "tool_call_id", None))
        }
        def call_ids(m) -> list:
            return [tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    for tc in (getattr(m, "tool_calls", None) or [])]

        # Drop requests missing any of their results.
        kept = [m for m in messages
                if not (isinstance(m, AIMessage) and call_ids(m)
                        and not all(i in answered for i in call_ids(m)))]
        # A result whose request we just dropped is now an orphan.
        surviving = {i for m in kept if isinstance(m, AIMessage) for i in call_ids(m)}
        return [m for m in kept
                if not (isinstance(m, ToolMessage)
                        and getattr(m, "tool_call_id", None) not in surviving)]

    @staticmethod
    def _latest_answer(messages: list) -> str:
        """Most recent model-authored answer text.

        Only an AIMessage counts. Skipping non-AI messages matters: a REDIRECT or
        FORCE_FINALIZE appends a `[governor:...]` HumanMessage, and returning that
        would print the Governor's instruction to the user as if the agent had
        written it. Bare tool-call turns and raw tool output carry no answer.
        """
        for m in reversed(messages):
            if not isinstance(m, AIMessage) or getattr(m, "tool_calls", None):
                continue
            content = getattr(m, "content", "") or ""
            if isinstance(content, str) and content.strip():
                return content
        return ""


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
        # Normalize: use word-boundary matching and prefer the tool name that
        # appears earliest in the reply (reflects the LLM's lead choice).
        hits = []
        for n in self._tool_names:
            m = re.search(rf"\b{re.escape(n)}\b", answer)
            if m:
                hits.append((m.start(), n))
        if hits:
            hits.sort()  # earliest mention in the reply wins
            return hits[0][1]
        return None


def make_tool_choice_run(query: str):
    """Build the (judge, executable, tools) triple for the tool-choice demo,
    shared by main() and the integration test so both wire it identically."""
    tools = [web_search, wiki_lookup]
    judge = ToolChoiceJudge(
        query=query, tool_names=[t.name for t in tools])
    executable = SearchAgentExecutable(tools=tools)
    return judge, executable, tools


# ─── Demo judge (showcases steering deterministically) ───────────────────────
class DemoSteeringJudge:
    """A TrajectoryJudge that fires corrections on a fixed schedule so the
    Governor's steering is visible on every run — regardless of what the agent
    actually does.

    Emits REDIRECT after the 1st step (a recorded, dashboard-visible nudge),
    then STOP after the 2nd step, which raises ExecutionAborted in the
    executable and drives the run to ExecutionStatus.ABORTED.
    """

    def judge(self, trajectory):
        n = len(trajectory)
        if n == 1:
            return Correction(
                flag=CorrectionFlag.REDIRECT,
                note="demo: consider a more specific query",
                step_index=n - 1,
            )
        if n >= 2:
            return Correction(
                flag=CorrectionFlag.STOP,
                note="demo: governor halting the run",
                step_index=n - 1,
            )
        return Correction(flag=CorrectionFlag.CONTINUE)


# ─── A user-defined business judge ───────────────────────────────────────────
def agent_text_from_step(step) -> str:
    """The model-authored prose in a step, read off its raw streamed chunk.

    `step.summary` is only the graph node name ("agent" / "tools") — the actual
    response text lives in the chunk, same shape `tool_name_from_step` walks:
    {node: {"messages": [AIMessage, ...]}}. Tool output (ToolMessage, which has
    a .name) and bare tool-call turns carry no answer text, so both are skipped.
    """
    raw = getattr(step, "raw", None)
    if not isinstance(raw, dict):
        return ""
    parts = []
    for node_out in raw.values():
        msgs = node_out.get("messages") if isinstance(node_out, dict) else None
        if not msgs:
            continue
        for m in msgs:
            if getattr(m, "name", None):        # ToolMessage: raw tool output
                continue
            if getattr(m, "tool_calls", None):  # a pending tool call, not prose
                continue
            content = getattr(m, "content", "") or ""
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):     # multimodal content blocks
                parts.extend(b.get("text", "") for b in content
                             if isinstance(b, dict))
    return "".join(parts)


# Judges are plain functions. This one is the shape a business metric takes:
# read what you need off the context, return a correction or None.
def make_answer_length_cap(max_chars: int):
    """A cost judge, parameterized: stop paying for search rounds once the answer
    is already long enough to be useful.

    Measures the real response text off each step's raw chunk. Reading
    `s.summary` instead would only ever count the 5-character node name, so the
    cap could never be reached inside the co-registered step budget.
    """

    @judge(name="answer_length_cap")
    def answer_length_cap(ctx):
        chars = sum(len(agent_text_from_step(s)) for s in ctx if s.kind == "agent")
        if chars > max_chars:
            return finalize("answer is long enough; wrap up",
                            meta={"agent_chars": chars, "cap": max_chars})
        return None

    return answer_length_cap


# A second business judge, to show two custom metrics coexisting with the
# built-ins. It returns REDIRECT — a LOWER severity than the length cap's
# FORCE_FINALIZE — so when both fire on the same step you can watch the Governor
# push the finalize and record this one as having lost arbitration.
def make_search_budget(max_searches: int):
    """A spend judge: each tool call costs a search API request, so cap them and
    make the agent answer from what it already retrieved.

    Counts executed tool steps via `tool_name_from_step`, which reads the tool
    name off the raw chunk. This is reachable MID-run — unlike a judge that only
    inspects the final answer, which the run ends before the Governor can see.
    """

    @judge(name="search_budget")
    def search_budget(ctx):
        used = sum(1 for s in ctx
                   if s.kind == "tool" and tool_name_from_step(s) is not None)
        if used >= max_searches:
            # Live counts belong in meta, not the note: the Governor de-duplicates
            # a judge's repeat verdict on (flag, note, step_index), so a note that
            # changes every poll would push a fresh correction each time.
            return redirect(
                f"search budget spent ({max_searches} max); "
                "answer from the results you already have",
                step_index=len(ctx) - 1,
                meta={"searches": used, "budget": max_searches})
        return None

    return search_budget


# ─── Main ────────────────────────────────────────────────────────────────────
async def main():
    argv = sys.argv[1:]
    demo_steering = "--demo-steering" in argv
    tool_choice = "--tool-choice" in argv
    tight = "--tight-budgets" in argv
    argv = [a for a in argv
            if a not in ("--demo-steering", "--tool-choice", "--tight-budgets")]
    if tool_choice:
        default_query = "Find information about PEP 703 and summarize it."
    elif tight:
        # Open-ended on purpose: the agent searches, then writes at length, so
        # the shrunk budgets below are actually reached.
        default_query = ("Explain Python 3.14 free threading, the GIL removal "
                         "work, and what it means for multi-threaded programs.")
    else:
        default_query = "What is Python 3.14 free threading?"
    query = " ".join(argv) if argv else default_query

    if demo_steering:
        judges = [DemoSteeringJudge()]
        executable = SearchAgentExecutable()
    elif tool_choice:
        tool_judge, executable, _ = make_tool_choice_run(query)
        judges = [tool_judge]
    else:
        # The curated built-ins composed with two user-defined business judges.
        # --tight-budgets keeps the SAME roster and only shrinks the thresholds,
        # so a well-behaved agent trips them and you can watch the mechanism.
        #
        # Which judges fire under --tight-budgets, and why:
        #   search_budget    REDIRECT       after the 1st tool call — the agent is
        #                                   steered mid-run and you see it comply
        #   step_budget      FORCE_FINALIZE once the redirect pushes past 4 steps
        #   latency_budget   generous (60s) on purpose: a 2s cap fires before the
        #                    agent does anything and, being the most severe flag,
        #                    masks every other judge.
        #   repeat_tool / output_loop  usually silent — they detect MISBEHAVIOUR,
        #     and a healthy run alternates agent/tool instead of repeating either.
        #     Use --demo-steering to see corrections on a fixed schedule.
        judges = [
            StepBudgetJudge(max_steps=4 if tight else 20),
            RepeatToolJudge(threshold=2 if tight else 3),
            OutputLoopJudge(window=2 if tight else 3),
            LatencyBudgetJudge(max_seconds=60 if tight else 120),
            make_answer_length_cap(400 if tight else 2000),
            make_search_budget(1 if tight else 6),
        ]
        executable = SearchAgentExecutable()

    print(f"\n{'='*60}")
    print(f"  Sentinal Search Agent")
    print(f"  Query: {query}")
    print(f"  Thread: {threading.current_thread().native_id} (main)")
    print(f"  GIL: {'enabled' if sys._is_gil_enabled() else 'DISABLED'}")
    if demo_steering:
        judge_desc = "forces REDIRECT then STOP"
    elif tool_choice:
        judge_desc = "steers wrong tool → right tool"
    elif tight:
        judge_desc = "TIGHT BUDGETS — these will fire"
    else:
        judge_desc = "fire only if the agent misbehaves"
    print(f"  Judges ({judge_desc}):")
    for j in judges:
        name = getattr(j, "name", type(j).__name__)
        # @judge copies the decorated function's __module__ onto the instance, so
        # this reads "custom" for judges defined here and "built-in" for the
        # curated ones shipped in sentinal.judges.
        origin = ("built-in" if getattr(j, "__module__", "").startswith("sentinal")
                  else "custom")
        print(f"    - {name:<20} [{origin}]")
    print(f"{'='*60}\n")

    # Create runtime and session. The judges flow through execute(), so there is
    # no hand-registered Governor to keep in sync.
    runtime = SentinelRuntime()
    session = ExecutionSession(
        executable=executable,
        context=ExecutionContext(input={"query": query}),
    )

    # Execute with full 3-thread supervision
    t0 = time.perf_counter()
    if dashboard_enabled():
        dash = LiveThreadDashboard()
        result, _ = await asyncio.gather(
            runtime.execute(session, judges=judges),
            dash.run(session),
        )
    else:
        result = await runtime.execute(session, judges=judges)
    elapsed = time.perf_counter() - t0

    # Extract result
    memory = result.state.get_all_memory()
    agent_result = memory.get("result", {})

    # Governor steering summary
    governor = next((a for a in runtime.actors if a.ref.name == "governor"), None)
    interventions = governor.get_interventions() if governor else []
    corrections = [i for i in interventions if i.get("type") == "correction"]

    print(f"\n{'─'*60}")
    print(f"  Response:")
    print(f"{'─'*60}")
    print(f"\n{agent_result.get('response', 'No response') or '(run aborted before a final answer)'}\n")
    print(f"{'─'*60}")
    print(f"  Stats:")
    print(f"    Duration: {elapsed*1000:.0f}ms")
    print(f"    Final status: {result.state.status.value}")
    print(f"    Trajectory steps: {agent_result.get('steps', len(result.get_trajectory()))}")
    print(f"    Worker thread: {agent_result.get('thread_id')}")
    print(f"    Main thread: {threading.current_thread().native_id}")
    print(f"{'─'*60}")
    pushed = [c for c in corrections if c.get("pushed", True)]
    print(f"  Governor findings: {len(corrections)}"
          f"  ({len(pushed)} pushed to the agent, "
          f"{len(corrections) - len(pushed)} lost arbitration)")
    for c in corrections:
        note = f" — {c['note']}" if c.get("note") else ""
        mark = "→" if c.get("pushed", True) else "·"
        tail = "" if c.get("pushed", True) else "  (a more severe correction won)"
        print(f"    {mark} [{c.get('judge', '?')}] {c['flag']}{note}{tail}")
        if c.get("meta"):
            print(f"        meta: {c['meta']}")
    if not corrections:
        print("    (none — the agent stayed inside every budget)")
    if memory.get("abort_reason"):
        print(f"  Aborted: {memory['abort_reason']}")
    print(f"{'─'*60}\n")

    await runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())