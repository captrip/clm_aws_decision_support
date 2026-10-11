# Governor Tool-Choice Steering — Design

**Date:** 2026-07-28
**Status:** Approved (brainstorming)
**Scope:** Extend `examples/search_agent.py` with a watchable demo where the
Governor detects a wrong tool choice and steers the agent to the right tool,
plus an integration test with soft assertions.

## Goal

Make the Governor's tool-choice correction visible and testable: the agent is
given two tools, reaches for the wrong one for a given query, and the Governor
issues an `AVOID_TOOL` correction that guides it to the right tool. This
exercises the existing `AVOID_TOOL` steering path (which currently only ever
fires on a repeated-tool loop) with a semantic, LLM-judged wrong-tool signal.

## Scenario & Tools

Add a second tool to `examples/search_agent.py` alongside the existing
`web_search` (DuckDuckGo via `ddgs`):

- **`wiki_lookup(topic: str)`** — queries the real Wikipedia REST summary API
  (`https://en.wikipedia.org/api/rest_v1/page/summary/{title}`) using stdlib
  `urllib.request` (no new dependency). Returns the article extract plus the
  canonical page URL. On network/HTTP error, returns a short error string
  (the tool never raises — it degrades gracefully like `web_search`).
- Both tools are bound to the agent when running the tool-choice scenario.

A new CLI flag `--tool-choice` (mirroring the existing `--demo-steering` flag)
runs with a wiki-favoring default query:

> "Look up the Wikipedia article on PEP 703."

The agent is tempted to reach for general `web_search`; the Governor's judge
catches the mismatch and steers it to `wiki_lookup`.

## Judge — `ToolChoiceJudge` (LLM-based)

A new judge class defined in `examples/search_agent.py`, next to
`DemoSteeringJudge` (demo-specific, and it needs LLM access). Implements the
`TrajectoryJudge` protocol: `judge(trajectory) -> Correction`.

Logic:

1. Inspect the **latest step**. If it is not a fresh tool call (no new tool
   invocation since the last judged step), return `CONTINUE` — no LLM call.
   This keeps the judge cheap and avoids judging every agent step.
2. Extract the tool name the agent just called (from the streamed chunk's
   tool-call metadata; fall back to the step summary/node name).
3. If this exact wrong-tool episode was already corrected, return `CONTINUE`
   (dedupe guard — one correction per wrong-tool episode, so we do not repeat
   the same correction on every poll).
4. Otherwise make **one small, tool-less LLM call** (temperature 0):
   > Given the user query `<q>` and available tools `[web_search, wiki_lookup]`,
   > the agent chose `<tool>`. Reply with the single best tool name, or `ok` if
   > the choice is fine.
5. If the LLM names a **different** tool, return
   `Correction(flag=AVOID_TOOL, note="<wrong_tool>; use <right_tool> instead")`
   and record the episode as corrected. Else return `CONTINUE`.

The judge reuses the demo's `get_llm()` factory. If the LLM call raises, the
judge returns `CONTINUE` (a broken judge must never crash the run — the Governor
already wraps `judge()` in a try/except, and we belt-and-suspenders here too).

### Executable interaction (already implemented)

The executable already handles `AVOID_TOOL` by appending a
`[governor:avoid_tool] <note>` message and restarting the stream, so the agent
sees the steer and re-picks a tool. No change needed to the correction-handling
loop. The grounding system prompt stays in effect.

## Test — `tests/test_trajectory/test_tool_choice_steering.py`

A real integration test using the real `ToolChoiceJudge` (real LLM), wired
exactly like the `--tool-choice` demo, through the real runtime.

Soft assertions (resilient to LLM variance):

1. The Governor issued **at least one** `AVOID_TOOL` correction whose note names
   `web_search`.
2. `wiki_lookup` appears in the trajectory (the agent switched to the right tool
   after being steered).
3. The run **completed** (status not aborted/errored).

Skip conditions (so CI without external access does not hard-fail):

- Skip if LLM credentials/provider are unavailable (catch provider/auth error
  during `get_llm()` or first call → `pytest.skip`).
- Skip if the Wikipedia API / network is unreachable.

The test is expected to be slow (real LLM + network) and may be marked
accordingly, but asserts loosely rather than on exact strings.

## Files Touched

- `examples/search_agent.py`
  - Add `wiki_lookup` tool (stdlib `urllib`).
  - Register both tools for the tool-choice scenario; `build_agent()` gains an
    optional tool set so the default single-tool behavior is preserved.
  - Add `ToolChoiceJudge`.
  - Add `--tool-choice` flag + default query + help text.
- `tests/test_trajectory/test_tool_choice_steering.py` — new test.
- No changes to core `sentinal/` — the `AVOID_TOOL` machinery already exists.

## What the User Sees

Running:

```bash
PYTHON_GIL=0 .ft-env/bin/python examples/search_agent.py --tool-choice
```

Live dashboard sequence:

1. Agent calls `web_search`.
2. Governor row flips to `correction · avoid_tool:web_search; use wiki_lookup`.
3. Agent's next step calls `wiki_lookup`.
4. Run completes; the end-of-run "Governor corrections issued" block lists the
   tool-choice correction.

## Non-Goals

- No changes to `HeuristicJudge` or its repeated-tool `AVOID_TOOL` rule.
- No deterministic/regex tool-choice judge (explicitly chose LLM-based).
- No new third-party dependencies (Wikipedia via stdlib `urllib`).
