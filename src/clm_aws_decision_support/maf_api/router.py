"""FastAPI router for the MAF Studio (lifecycle, catalogues, governance, AG-UI)."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from ag_ui.core import EventType, RunAgentInput, RunStartedEvent
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..auth import SupabaseUser, required_supabase_user
from . import agui, service
from .models import (
    CreateRequestBody,
    Evaluation,
    EvaluationProps,
    MemberBody,
    RequestDoc,
    ReviewBody,
    Role,
    SubmitAgentBody,
    SubmitWorkflowBody,
)
from .service import Actor
from .store import DomainError, MafStore, get_maf_store, now

T = TypeVar("T")
RunnerFactory = Callable[[Request, str | None, EvaluationProps], agui.Runner]


def default_runner_factory(request: Request, api_key: str | None, props: EvaluationProps) -> agui.Runner:
    """Evaluation runner backed by the MAF + Sentinel runtime (imported lazily)."""
    try:
        from integration.conversational_demo import run_conversational_demo
        from sentinal import MonaClient
    except ImportError as error:
        raise HTTPException(
            status_code=503,
            detail="The MAF/Sentinel runtime is not installed. Install the products/maf and products/sentinel packages.",
        ) from error

    base_url = os.environ.get("MONA_API_URL", str(request.base_url).rstrip("/"))
    client = MonaClient(api_key=api_key, base_url=base_url)

    async def runner(question: str, agent_name: str, instructions: str, on_event: Any) -> dict[str, Any]:
        return await run_conversational_demo(
            client=client,
            question=question,
            agent_name=agent_name,
            instructions=instructions,
            conversation=[],
            policy=props.policy,
            budget=props.budget,
            quality_threshold=props.quality_threshold,
            on_event=on_event,
        )

    return runner


def get_runner_factory() -> RunnerFactory:
    """Dependency hook so tests can substitute a fake runner."""
    return default_runner_factory


def _admin_emails() -> set[str]:
    return {item.strip().lower() for item in os.environ.get("MAF_ADMIN_EMAILS", "").split(",") if item.strip()}


def _is_production() -> bool:
    return os.environ.get("CLM_RUNTIME_MODE", "demo").strip().lower() == "production"


def _guard(call: Callable[[], T]) -> T:
    try:
        return call()
    except DomainError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message) from error


def current_actor(
    user: SupabaseUser = Depends(required_supabase_user),
    store: MafStore = Depends(get_maf_store),
) -> Actor:
    if not user.email:
        raise HTTPException(status_code=403, detail="Your account has no email address.")
    email = user.email.strip().lower()
    roles = store.get_roles(email)
    if email in _admin_emails() and Role.ADMIN not in roles:
        store.set_roles(email, [*roles, Role.ADMIN], "system")
        roles = store.get_roles(email)
    elif not roles and not _is_production() and not store.has_admin():
        store.bootstrap_admin(email)
        roles = store.get_roles(email)
    return Actor(user_id=user.id, email=email, roles=frozenset(roles))


def member(actor: Actor = Depends(current_actor)) -> Actor:
    if not actor.roles:
        raise HTTPException(status_code=403, detail="You have not been assigned a MAF role. Ask an admin for access.")
    return actor


def _catalogue(store: MafStore) -> dict[str, Any]:
    return {
        "workflows": [item.model_dump(mode="json") for item in store.list_workflows()],
        "agents": [item.model_dump(mode="json") for item in store.list_agents()],
    }


def _state(store: MafStore) -> dict[str, Any]:
    return {"requests": [doc.public() for doc in store.list_requests()], "catalogue": _catalogue(store)}


def _detail(store: MafStore, doc: RequestDoc) -> dict[str, Any]:
    return doc.public(store.list_events(doc.id))


def _evaluation_instructions(doc: RequestDoc) -> str:
    parts = [f"Requirement: {doc.requirement}", f"Workflow steps: {doc.workflow.steps}"]
    parts += [f"Agent {agent.agent}: {agent.instructions}" for agent in doc.agents]
    return "\n".join(parts)[:2000]


def create_maf_router(*, api_key_dependency: Callable[..., Any]) -> APIRouter:
    router = APIRouter(prefix="/api/maf", tags=["maf"])

    @router.get("/me")
    def me(actor: Actor = Depends(current_actor)) -> dict[str, Any]:
        return {
            "email": actor.email,
            "roles": sorted(role.value for role in actor.roles),
            "separation_of_duties": service.separation_of_duties(),
        }

    @router.get("/members")
    def list_members(actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)) -> list[dict[str, Any]]:
        _guard(lambda: actor.require(Role.ADMIN))
        return store.list_members()

    @router.put("/members")
    def put_member(
        body: MemberBody, actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> list[dict[str, Any]]:
        _guard(lambda: actor.require(Role.ADMIN))
        _guard(lambda: store.set_roles(body.email, body.roles, actor.email))
        return store.list_members()

    @router.delete("/members/{email}")
    def delete_member(
        email: str, actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> list[dict[str, Any]]:
        _guard(lambda: actor.require(Role.ADMIN))
        _guard(lambda: store.set_roles(email, [], actor.email))
        return store.list_members()

    @router.get("/catalogue")
    def catalogue(_actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)) -> dict[str, Any]:
        return _catalogue(store)

    @router.get("/requests")
    def list_requests(_actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)) -> list[dict[str, Any]]:
        return [doc.public() for doc in store.list_requests()]

    @router.post("/requests", status_code=201)
    def create_request(
        body: CreateRequestBody, actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> dict[str, Any]:
        return _detail(store, _guard(lambda: service.create_request(store, actor, body)))

    @router.get("/requests/{request_id}")
    def get_request(
        request_id: str, _actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> dict[str, Any]:
        doc = store.get_request(request_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Request not found.")
        return _detail(store, doc)

    @router.post("/requests/{request_id}/workflow/submit")
    def submit_workflow(
        request_id: str, body: SubmitWorkflowBody, actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> dict[str, Any]:
        return _detail(store, _guard(lambda: service.submit_workflow(store, actor, request_id, body.steps)))

    @router.post("/requests/{request_id}/workflow/review")
    def review_workflow(
        request_id: str, body: ReviewBody, actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> dict[str, Any]:
        return _detail(
            store, _guard(lambda: service.review_workflow(store, actor, request_id, body.approve, body.note))
        )

    @router.post("/requests/{request_id}/agents/{agent}/submit")
    def submit_agent(
        request_id: str,
        agent: str,
        body: SubmitAgentBody,
        actor: Actor = Depends(member),
        store: MafStore = Depends(get_maf_store),
    ) -> dict[str, Any]:
        return _detail(store, _guard(lambda: service.submit_agent(store, actor, request_id, agent, body.instructions)))

    @router.post("/requests/{request_id}/agents/{agent}/review")
    def review_agent(
        request_id: str,
        agent: str,
        body: ReviewBody,
        actor: Actor = Depends(member),
        store: MafStore = Depends(get_maf_store),
    ) -> dict[str, Any]:
        return _detail(
            store, _guard(lambda: service.review_agent(store, actor, request_id, agent, body.approve, body.note))
        )

    @router.post("/requests/{request_id}/release/review")
    def review_release(
        request_id: str, body: ReviewBody, actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> dict[str, Any]:
        return _detail(
            store, _guard(lambda: service.review_release(store, actor, request_id, body.approve, body.note))
        )

    @router.post("/requests/{request_id}/handover")
    def handover(
        request_id: str, actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> dict[str, Any]:
        return _detail(store, _guard(lambda: service.hand_over(store, actor, request_id)))

    @router.post("/requests/{request_id}/agui")
    async def run_evaluation(
        request_id: str,
        body: RunAgentInput,
        request: Request,
        actor: Actor = Depends(member),
        principal: Any = Depends(api_key_dependency),
        store: MafStore = Depends(get_maf_store),
        factory: RunnerFactory = Depends(get_runner_factory),
    ) -> StreamingResponse:
        if getattr(principal, "user_id", None) != actor.user_id:
            raise HTTPException(status_code=403, detail="The API key does not belong to the signed-in user.")
        doc = _guard(lambda: service.check_can_evaluate(actor, store.get_request(request_id)))
        forwarded = body.forwarded_props if isinstance(body.forwarded_props, dict) else {}
        props = EvaluationProps.model_validate(forwarded)
        prompt = ""
        for message in reversed(body.messages):
            content = getattr(message, "content", None)
            if getattr(message, "role", None) == "user" and isinstance(content, str) and content.strip():
                prompt = content.strip()
                break
        if not prompt:
            raise HTTPException(status_code=422, detail="Send a user message to evaluate.")
        runner = factory(request, request.headers.get("X-API-Key"), props)
        instructions = _evaluation_instructions(doc)

        def on_complete(result: dict[str, Any]) -> dict[str, Any]:
            sentinel = result.get("sentinel") or {}
            evaluation = Evaluation(
                execution_id=str(result.get("execution_id") or f"exec_{uuid.uuid4().hex[:10]}"),
                status=str(sentinel.get("status", "completed")),
                model=str(result.get("model", "")),
                provider=result.get("provider"),
                policy=result.get("policy"),
                answer=str(result.get("answer", "")),
                prompt=prompt,
                cost=float(result.get("cost") or 0),
                elapsed_ms=float(result.get("elapsed_ms") or 0),
                decision_id=result.get("decision_id"),
                quality=list(sentinel.get("quality") or []),
                trace=list(sentinel.get("trace") or []),
                run_by=actor.email,
                run_at=now(),
            )
            updated = _guard(lambda: service.record_evaluation(store, actor, request_id, evaluation))
            return {"phase": "complete", "request": updated.public()}

        stream = agui.evaluation_stream(
            thread_id=body.thread_id or request_id,
            run_id=body.run_id or f"run_{uuid.uuid4().hex[:10]}",
            question=prompt,
            agent_name=doc.name,
            instructions=instructions,
            runner=runner,
            on_complete=on_complete,
        )
        return StreamingResponse(stream, media_type=agui.SSE_MEDIA_TYPE, headers=agui.sse_headers())

    @router.get("/stream")
    async def stream(
        request: Request, _actor: Actor = Depends(member), store: MafStore = Depends(get_maf_store)
    ) -> StreamingResponse:
        """Live AG-UI feed: a STATE_SNAPSHOT on connect and after every change."""
        enc = agui.encoder()

        async def frames() -> AsyncIterator[str]:
            yield enc.encode(
                RunStartedEvent(type=EventType.RUN_STARTED, thread_id="maf-studio", run_id=f"feed_{uuid.uuid4().hex[:8]}")
            )
            seen = -1
            idle = 0
            while not await request.is_disconnected():
                version = await asyncio.to_thread(store.version)
                if version != seen:
                    seen = version
                    idle = 0
                    state = await asyncio.to_thread(_state, store)
                    yield enc.encode(agui.snapshot_event(state))
                else:
                    idle += 1
                    if idle >= 15:
                        idle = 0
                        yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(frames(), media_type=agui.SSE_MEDIA_TYPE, headers=agui.sse_headers())

    return router
