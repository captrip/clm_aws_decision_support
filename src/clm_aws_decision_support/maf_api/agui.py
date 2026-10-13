"""AG-UI event stream for a MAF evaluation run, built on the official ag-ui-protocol SDK."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from ag_ui.core import (
    CustomEvent,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder

# runner(question, agent_name, instructions, on_event) -> demo result dict
Runner = Callable[[str, str, str, Callable[[dict[str, Any]], Awaitable[None]]], Awaitable[dict[str, Any]]]

SSE_MEDIA_TYPE = "text/event-stream"
_DONE = object()


def encoder() -> EventEncoder:
    return EventEncoder()


def sse_headers() -> dict[str, str]:
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def snapshot_event(state: dict[str, Any]) -> StateSnapshotEvent:
    return StateSnapshotEvent(type=EventType.STATE_SNAPSHOT, snapshot=state)


async def evaluation_stream(
    *,
    thread_id: str,
    run_id: str,
    question: str,
    agent_name: str,
    instructions: str,
    runner: Runner,
    on_complete: Callable[[dict[str, Any]], dict[str, Any]],
) -> AsyncIterator[str]:
    """Yield AG-UI SSE frames while ``runner`` executes.

    ``on_complete`` persists the result and returns the final shared state.
    """
    enc = encoder()
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def on_event(event: dict[str, Any]) -> None:
        await queue.put(event)

    async def run() -> dict[str, Any]:
        try:
            return await runner(question, agent_name, instructions, on_event)
        finally:
            await queue.put(_DONE)

    yield enc.encode(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_id))
    yield enc.encode(
        snapshot_event({"phase": "running", "stage": "starting", "agent": agent_name, "attempts": 0, "quality": None})
    )

    task = asyncio.create_task(run())
    tool_call_id: str | None = None
    attempts = 0
    try:
        while True:
            event = await queue.get()
            if event is _DONE:
                break
            stage = event.get("stage")
            status = event.get("status")
            if stage == "maf_agent" and status == "started":
                attempts += 1
                tool_call_id = f"call_{uuid.uuid4().hex[:10]}"
                yield enc.encode(
                    StepStartedEvent(type=EventType.STEP_STARTED, step_name=f"agent:{event.get('agent', agent_name)}#{attempts}")
                )
                yield enc.encode(
                    ToolCallStartEvent(type=EventType.TOOL_CALL_START, tool_call_id=tool_call_id, tool_call_name="mona_route")
                )
                yield enc.encode(
                    ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=tool_call_id,
                        delta=json.dumps({"agent": event.get("agent", agent_name), "attempt": attempts}),
                    )
                )
                yield enc.encode(ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=tool_call_id))
                yield enc.encode(
                    StateDeltaEvent(
                        type=EventType.STATE_DELTA,
                        delta=[
                            {"op": "replace", "path": "/stage", "value": "routing"},
                            {"op": "replace", "path": "/attempts", "value": attempts},
                        ],
                    )
                )
            elif stage == "maf_agent" and status == "completed":
                if tool_call_id:
                    yield enc.encode(
                        ToolCallResultEvent(
                            type=EventType.TOOL_CALL_RESULT,
                            message_id=f"msg_{uuid.uuid4().hex[:10]}",
                            tool_call_id=tool_call_id,
                            content=json.dumps(
                                {k: event.get(k) for k in ("model", "provider", "cost", "elapsed_ms", "decision_id") if k in event},
                                default=str,
                            ),
                            role="tool",
                        )
                    )
                yield enc.encode(
                    StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=f"agent:{event.get('agent', agent_name)}#{attempts}")
                )
                yield enc.encode(
                    StateDeltaEvent(
                        type=EventType.STATE_DELTA, delta=[{"op": "replace", "path": "/stage", "value": "evaluating"}]
                    )
                )
            elif stage in {"sentinel_evaluation", "sentinel_steering"}:
                yield enc.encode(
                    CustomEvent(
                        type=EventType.CUSTOM,
                        name=stage.replace("_", "."),
                        value=json.loads(json.dumps(event, default=str)),
                    )
                )
                if stage == "sentinel_evaluation":
                    yield enc.encode(
                        StateDeltaEvent(
                            type=EventType.STATE_DELTA,
                            delta=[{"op": "replace", "path": "/quality", "value": event.get("score", event.get("quality"))}],
                        )
                    )
        result = await task
        final_state = on_complete(result)
        message_id = f"msg_{uuid.uuid4().hex[:10]}"
        yield enc.encode(TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id=message_id, role="assistant"))
        answer = str(result.get("answer", ""))
        for index in range(0, len(answer), 80):
            yield enc.encode(
                TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT, message_id=message_id, delta=answer[index : index + 80]
                )
            )
        yield enc.encode(TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=message_id))
        yield enc.encode(snapshot_event(final_state))
        yield enc.encode(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_id))
    except asyncio.CancelledError:
        task.cancel()
        raise
    except Exception as error:  # noqa: BLE001 - surfaced to the client as RUN_ERROR
        if not task.done():
            task.cancel()
        yield enc.encode(RunErrorEvent(type=EventType.RUN_ERROR, message=str(error) or error.__class__.__name__))
