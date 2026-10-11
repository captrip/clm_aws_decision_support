# multi-threaded-agents
To create multi-threaded agents
uv venv --python 3.14t .ft-env
./.ft-env/bin/python -m ensurepip --default-pip
./.ft-env/bin/python -m pip install ipykernel
./.ft-env/bin/python -m ipykernel install --user --name="multi-thrd-agents" --display-name="Python 3.14 (Free-Threaded)" --env PYTHON_GIL 0

## Live thread dashboard

Watch each worker thread (Executor, Observer, Governor) act in real time,
one row per thread, updating live as the agent runs.

Enable it when configuring observability:

    from sentinal import configure
    configure(live_dashboard=True)

or via environment variable:

    export SENTINAL_LIVE_DASHBOARD=1

Install the optional dependency:

    pip install "sentinal[dashboard]"

The dashboard renders with [rich](https://github.com/Textualize/rich). If
`rich` isn't installed, or stdout isn't a TTY (piped/CI), it degrades
gracefully to periodic plain-text status lines. It is opt-in and off by
default, so normal runs are unaffected.

See `examples/search_agent.py` for a working integration.

## Governor trajectory correction

The Governor thread judges the streaming execution trajectory with a pluggable
`TrajectoryJudge` (default: `HeuristicJudge`). It pushes typed `Correction`
objects with flags:

- `REDIRECT`: steer the agent to a different approach
- `AVOID_TOOL`: prevent the agent from using a specific tool
- `FORCE_FINALIZE`: skip further tool calls and return a response
- `STOP`: halt execution immediately

The executable acts on corrections between steps. `STOP` yields
`ExecutionStatus.ABORTED`. Each correction appears in the live dashboard's
Governor row. The heuristic judge detects repeat loops, budget violations,
infinite tool chains, and premature continuation calls.

## Pluggable judges

Pass a list of judges to a single run. Every judge is asked on each new step,
every finding is recorded, and the **most severe** correction is pushed to the
agent — ties break by registration order:

    STOP > FORCE_FINALIZE > AVOID_TOOL > REDIRECT > CONTINUE

    from sentinal import StepBudgetJudge, LatencyBudgetJudge, judge, finalize

    @judge(name="cost_cap")
    def cost_cap(ctx):
        spend = ctx.memory.get("usd", 0.0)
        if spend > 2.0:
            return finalize(f"cost cap exceeded", meta={"spend": spend})
        return None    # None means "nothing to say"

    await runtime.execute(session, judges=[
        StepBudgetJudge(max_steps=20),
        LatencyBudgetJudge(max_seconds=120),
        cost_cap,
    ])

Judges are structurally typed — any object with a `.judge(ctx)` method works, so
`@judge` is a convenience, not a requirement. A judge that raises is logged and
skipped; it cannot crash the run or suppress its peers. `judges=` is scoped to
one session, so it never leaks into a later run or races a concurrent one.

**Built-in judges:** `StepBudgetJudge` (runaway loop → `FORCE_FINALIZE`),
`RepeatToolJudge` (same tool repeatedly → `AVOID_TOOL`), `OutputLoopJudge`
(model repeating itself → `REDIRECT`), `LatencyBudgetJudge` (wall-clock cap →
`FORCE_FINALIZE`, and it fires even on a stalled trajectory).

**Correction constructors:** `redirect()`, `avoid_tool()`, `finalize()`,
`stop()` — each takes an optional `note` (the guidance the agent is shown) and
`meta` (structured data for your logs, kept out of the agent's prompt).

Keep the `note` **stable** while a condition holds. The Governor de-duplicates a
judge's repeat verdict on `(flag, note, step_index)`, so interpolating a live
counter or clock into the note defeats that and pushes one correction per poll.
Put changing values in `meta` instead.

## Running the example

`examples/search_agent.py` is a live LangGraph ReAct agent with web search,
running under full 3-thread supervision. It has four modes.

**`PYTHONPATH=.` is required** — an installed `sentinal` in site-packages
otherwise shadows this repo and the judge API won't import.

Pick a provider first:

    # Azure OpenAI
    export LLM_PROVIDER=azure
    export AZURE_OPENAI_API_KEY="your-key"
    export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
    export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
    export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

    # or AWS Bedrock (the default; needs valid SSO credentials)
    export LLM_PROVIDER=aws
    export AWS_PROFILE="your-profile"

A `.env` file in the repo root is loaded automatically. Add
`SENTINAL_LIVE_DASHBOARD=1` to any mode below for the live 3-thread view.

### 1. `--tight-budgets` — see the judges fire (start here)

    PYTHON_GIL=0 PYTHONPATH=. .ft-env/bin/python examples/search_agent.py --tight-budgets

Same judge roster as a production run, thresholds shrunk so a *well-behaved*
agent trips them. Prints the roster tagged built-in vs custom, then every
finding — which judge raised it, its `meta`, and whether it was pushed (`→`) or
lost arbitration (`·`):

    Governor findings: 5  (3 pushed to the agent, 2 lost arbitration)
      → [search_budget] redirect — search budget spent (1 max); answer from ...
          meta: {'searches': 1, 'budget': 1}
      → [answer_length_cap] force_finalize — answer is long enough; wrap up
          meta: {'agent_chars': 846, 'cap': 400}
      · [search_budget] redirect — search budget spent (1 max); answer from ...
          meta: {'searches': 1, 'budget': 1}   (a more severe correction won)

That third line is severity arbitration in one place: `search_budget` and
`answer_length_cap` both fired on the same step, and `FORCE_FINALIZE` outranks
`REDIRECT`.

`repeat_tool` and `output_loop` usually stay silent even here — they detect
*misbehaviour*, and a healthy run alternates agent/tool rather than repeating
either. Use `--demo-steering` for a deterministic firing of every flag.

### 2. Normal run — production budgets

    PYTHON_GIL=0 PYTHONPATH=. .ft-env/bin/python examples/search_agent.py \
        "What is Python 3.14 free threading?"

The same six judges at realistic thresholds (20 steps, 120s, 6 searches, 2000
chars). Expect **zero** findings on a healthy run — that's the point: judges are
guardrails, not a processing stage. Any positional argument is the query;
omitting it uses a default.

### 3. `--demo-steering` — deterministic steering and abort

    PYTHON_GIL=0 PYTHONPATH=. .ft-env/bin/python examples/search_agent.py \
        --demo-steering "Tell me about asyncio"

A `DemoSteeringJudge` issues `REDIRECT` after step 1 and `STOP` after step 2,
regardless of what the agent does. Use it to watch the Governor intervene and
then abort: the run ends `Final status: aborted` with an `Aborted:` reason. This
mode is a fixed schedule, so it works without a well-behaved agent.

### 4. `--tool-choice` — an LLM-backed judge

    PYTHON_GIL=0 PYTHONPATH=. .ft-env/bin/python examples/search_agent.py --tool-choice

Gives the agent two tools (`web_search`, `wiki_lookup`) and registers a
`ToolChoiceJudge` that asks an LLM whether the agent picked the right one. On a
bad pick it issues `AVOID_TOOL` naming the better tool, at most once per wrong
tool. Findings depend on what the model actually chooses, so a clean run is a
legitimate outcome.

### Writing your own judge

The two business judges in the example — `make_answer_length_cap` and
`make_search_budget` — are the pattern to copy: read what you need off the
context, return a correction or `None`. `JudgeContext` is sequence-like, so
`for step in ctx`, `len(ctx)`, and `ctx[-3:]` all work, alongside
`ctx.elapsed_s`, `ctx.memory`, and `ctx.input`.

One trap worth knowing: a judge that only inspects the **final** answer text can
never fire, because the run ends before the Governor polls again. Judge
something reachable mid-run — step counts, tool calls, accumulated text.

### Flag precedence

`--demo-steering` wins over `--tool-choice`, which wins over `--tight-budgets`.
Each mode installs its own judge roster, so combining them isn't meaningful.

## Route model requests through MONA

Sentinel includes a small synchronous and async client for the Whitebat MONA
routing API. MONA selects and invokes a model according to the requested
policy; the API key is sent as `X-API-Key`.

Install Sentinel, then configure the API URL and a key created in the Whitebat
account portal:

```sh
python -m pip install "git+https://github.com/captrip/multi-threaded-agents.git"
export MONA_API_URL="https://your-mona-api.example"
export MONA_API_KEY="your-whitebat-api-key"
```

Use the client directly or from an async agent without blocking its event loop:

```python
from sentinal import MonaClient

mona = MonaClient()
result = await mona.ask_async(
    "Summarize the issue and propose next steps.",
    policy="linucb",
    budget=0.01,
)
print(result["model"], result["answer"])
```

Supported policy names are `linucb`, `clm`, and `hybrid`. API keys grant access
to the MONA account and should be stored in a secret manager or environment
variables, never in source code or logs. MONA response metadata includes the
selected model, policy, provider, usage, and cost.

The Whitebat hosted Sentinel walkthrough is a scripted product demonstration.
It does not execute arbitrary code or replace this local Python runtime.
