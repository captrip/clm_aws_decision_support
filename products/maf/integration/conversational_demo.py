"""Interactive MAF agent demo with MONA routing and Sentinel steering."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from MAFI.agent_factory.base_MAFI_agent import BaseMAFIAgent
from MAFI.agent_factory.types.state_manager import StateManager
from sentinal import (
    CorrectionFlag,
    JudgeContext,
    MonaClient,
    TrajectoryStep,
    redirect,
)
from sentinal.core.execution.context import ExecutionContext
from sentinal.core.execution.result import ExecutionResult
from sentinal.core.execution.session import ExecutionSession
from sentinal.core.executable import Executable
from sentinal.runtime.runtime import SentinelRuntime

DemoPolicy = Literal["linucb", "clm", "hybrid"]
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class MonaChatResponse:
    content: str
    routed: dict[str, Any]


class MonaChatAdapter:
    """Adapt MONA's routed answer to the minimal chat-model interface MAF uses."""

    def __init__(
        self,
        client: MonaClient,
        *,
        policy: DemoPolicy,
        budget: float | None,
    ) -> None:
        self.client = client
        self.policy = policy
        self.budget = budget

    async def ainvoke(self, messages: list[Any]) -> MonaChatResponse:
        prompt = "\n\n".join(
            f"{getattr(message, 'type', 'user').upper()}:\n"
            f"{getattr(message, 'content', str(message))}"
            for message in messages
        )
        routed = await self.client.ask_async(
            prompt,
            policy=self.policy,
            budget=self.budget,
        )
        answer = routed.get("answer")
        if not isinstance(answer, str):
            raise ValueError("MONA response did not contain a text answer.")
        return MonaChatResponse(content=answer, routed=routed)


class ResponseQualityJudge:
    """Score the first draft and steer weak drafts toward a useful revision."""

    name = "maf_demo_response_quality"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.evaluations: list[dict[str, Any]] = []

    def judge(self, ctx: JudgeContext):
        if len(ctx.trajectory) != 1:
            return None
        step = ctx.trajectory[0]
        if step.kind != "agent":
            return None

        content = step.raw.get("content")
        if not isinstance(content, str):
            return None

        quality = min(len(content.strip()) / 400, 1.0)
        self.evaluations.append(
            {
                "score": round(quality, 2),
                "threshold": self.threshold,
                "characters": len(content.strip()),
                "step": step.index,
            }
        )
        if quality < self.threshold:
            return redirect(
                "Improve the draft: answer the user's request directly, add "
                "useful detail, and finish with a concrete next step.",
                step_index=step.index,
                meta={"quality": quality, "threshold": self.threshold},
            )
        return None


class MAFConversationalWorkflow(Executable[dict[str, Any], dict[str, Any]]):
    """Run a MAF-built agent through Sentinel's runtime and Governor."""

    def __init__(
        self,
        *,
        agent: BaseMAFIAgent,
        query: str,
        conversation: list[dict[str, str]],
        max_steering_passes: int,
        judge: ResponseQualityJudge,
        on_event: EventCallback | None = None,
    ) -> None:
        self.on_event = on_event
        self.agent = agent
        self.query = query
        self.conversation = conversation
        self.max_steering_passes = max_steering_passes
        self.judge = judge

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            await self.on_event(event)

    async def _record(self, trace: list[dict[str, Any]], event: dict[str, Any]) -> None:
        trace.append(event)
        await self._emit(event)

    async def execute(
        self,
        context: ExecutionContext,
    ) -> ExecutionResult[dict[str, Any]]:
        session = context.session
        if session is None:
            raise RuntimeError("MAF demo requires Sentinel execution context.")

        conversation_text = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in self.conversation
        )
        state = {
            "user_query": self.query,
            "conversation_context": conversation_text or "(new conversation)",
            "steering_instruction": "None yet; answer the user directly.",
        }
        trace: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None

        for attempt in range(self.max_steering_passes + 1):
            await self._emit(
                {
                    "stage": "maf_agent",
                    "status": "started",
                    "agent": self.agent.name,
                    "attempt": attempt + 1,
                }
            )
            started = time.perf_counter()
            agent_result = await self.agent.arun(state)
            raw_response = agent_result["raw"]
            routed = raw_response.routed
            content = agent_result["content"]
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            final = {
                "answer": content,
                "decision_id": routed.get("decision_id"),
                "model": routed.get("model"),
                "provider": routed.get("provider"),
                "policy": routed.get("policy"),
                "cost": routed.get("cost", 0),
                "usage": routed.get("usage", {}),
                "rationale": routed.get("rationale", ""),
                "elapsed_ms": elapsed_ms,
            }
            session.append_step(
                TrajectoryStep(
                    index=attempt,
                    kind="agent",
                    summary=f"{agent_result['agent']} draft {attempt + 1}",
                    raw={"content": content, "decision_id": routed.get("decision_id")},
                    ts=time.monotonic(),
                )
            )
            session.update_memory("latest_answer", content)
            await self._record(
                trace,
                {
                    "stage": "maf_agent",
                    "agent": agent_result["agent"],
                    "attempt": attempt + 1,
                    "status": "completed",
                    **final,
                }
            )

            if attempt == 0:
                deadline = time.monotonic() + 3.0
                while not self.judge.evaluations and time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                if not self.judge.evaluations:
                    raise RuntimeError("Sentinel Governor did not evaluate the draft.")

            corrections = session.drain_corrections()
            if not corrections:
                if attempt == 0:
                    await self._record(
                        trace,
                        {
                            "stage": "sentinel_evaluation",
                            **self.judge.evaluations[0],
                            "correction": None,
                        }
                    )
                break

            correction = corrections[-1]
            await self._record(
                trace,
                {
                    "stage": "sentinel_evaluation",
                    **self.judge.evaluations[0],
                    "correction": correction.flag.value,
                    "instruction": correction.note,
                }
            )
            if (
                correction.flag is not CorrectionFlag.REDIRECT
                or attempt >= self.max_steering_passes
            ):
                break

            state["steering_instruction"] = correction.note or ""
            session.update_memory("sentinel_steering", state["steering_instruction"])
            await self._record(
                trace,
                {
                    "stage": "sentinel_steering",
                    "status": "applied",
                    "instruction": state["steering_instruction"],
                }
            )

        if final is None:
            raise RuntimeError("MAF workflow finished without an agent response.")
        return ExecutionResult(
            output={
                **final,
                "agent": self.agent.name,
                "execution_id": str(session.execution_id),
                "sentinel": {
                    "status": session.state.status.value,
                    "quality": self.judge.evaluations,
                    "trace": trace,
                },
            }
        )


async def run_conversational_demo(
    *,
    client: MonaClient,
    question: str,
    agent_name: str,
    instructions: str,
    conversation: list[dict[str, str]],
    policy: DemoPolicy,
    budget: float | None,
    quality_threshold: float,
    max_steering_passes: int = 1,
    on_event: EventCallback | None = None,
) -> dict[str, Any]:
    model = MonaChatAdapter(client, policy=policy, budget=budget)
    agent = BaseMAFIAgent(
        name=agent_name,
        system_prompt=instructions,
        human_templete=(
            "Conversation so far:\n{conversation_context}\n\n"
            "User's latest request:\n{user_query}\n\n"
            "Sentinel steering instruction:\n{steering_instruction}"
        ),
        llm=model,
        tools=[],
        state_manager_=StateManager(),
        read_context_from_state_manager=False,
    )
    judge = ResponseQualityJudge(quality_threshold)
    workflow = MAFConversationalWorkflow(
        agent=agent,
        query=question,
        conversation=conversation[-10:],
        max_steering_passes=max_steering_passes,
        judge=judge,
        on_event=on_event,
    )
    session = ExecutionSession(
        executable=workflow,
        context=ExecutionContext(
            input={"user_query": question},
            metadata={"agent": agent_name},
        ),
    )
    runtime = SentinelRuntime()
    try:
        completed = await runtime.execute(session, judges=[judge])
    finally:
        await runtime.shutdown()

    result = completed.state.current.result
    if result is None or not isinstance(result.output, dict):
        error = completed.get_memory("error", "The supervised run did not return output.")
        raise RuntimeError(str(error))
    return result.output
