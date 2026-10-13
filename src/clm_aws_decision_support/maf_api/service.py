"""Lifecycle rules for MAF workflow and agent requests.

Every transition is validated against the persisted state, so the API cannot be
driven out of order regardless of what a client sends.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from .models import (
    AgentWork,
    CatalogueAgent,
    CatalogueWorkflow,
    CreateRequestBody,
    Evaluation,
    RequestDoc,
    Role,
    WorkflowWork,
    name_key,
)
from .store import DomainError, MafStore, now


@dataclass(frozen=True)
class Actor:
    user_id: str
    email: str
    roles: frozenset[Role]

    def require(self, role: Role) -> None:
        if role not in self.roles:
            raise DomainError(403, f"This action requires the {role.value} role.")


def separation_of_duties() -> bool:
    default = "1" if os.environ.get("CLM_RUNTIME_MODE", "demo").strip().lower() == "production" else "0"
    return os.environ.get("MAF_SEPARATION_OF_DUTIES", default).strip() not in {"0", "false", "no", ""}


def _no_self_review(actor: Actor, author: str | None, what: str) -> None:
    if separation_of_duties() and author and author.lower() == actor.email.lower():
        raise DomainError(403, f"Separation of duties: you cannot review {what} you submitted.")


def _note(body_note: str | None, approve: bool) -> str:
    note = (body_note or "").strip()
    if not approve and not note:
        raise DomainError(422, "A note is required when rejecting.")
    return note


def create_request(store: MafStore, actor: Actor, body: CreateRequestBody) -> RequestDoc:
    actor.require(Role.CHATBOT_BUILDER)
    workflow_hit = store.get_workflow(body.name)
    agents = []
    for agent in body.agents:
        hit = store.get_agent(agent)
        agents.append(
            AgentWork(agent=agent, status="published", from_catalogue=True, instructions=hit.instructions)
            if hit
            else AgentWork(agent=agent)
        )
    stamp = now()
    doc = RequestDoc(
        id=f"req_{uuid.uuid4().hex[:12]}",
        name=body.name,
        requirement=body.requirement,
        created_by=actor.email,
        created_at=stamp,
        updated_at=stamp,
        workflow=(
            WorkflowWork(status="published", from_catalogue=True, steps=workflow_hit.steps)
            if workflow_hit
            else WorkflowWork()
        ),
        agents=agents,
    )
    missing = sum(1 for agent in agents if not agent.from_catalogue)
    store.create_request(
        doc,
        actor.email,
        f"workflow {'found in catalogue' if workflow_hit else 'sent to Workflow Builder'}; "
        f"{len(agents) - missing} agent(s) found, {missing} sent to Agent Builder",
    )
    return doc


def submit_workflow(store: MafStore, actor: Actor, request_id: str, steps: str) -> RequestDoc:
    actor.require(Role.WORKFLOW_BUILDER)

    def change(_store: MafStore, doc: RequestDoc) -> tuple[str, str, str, str]:
        if doc.workflow.status != "todo":
            raise DomainError(409, "This workflow is not waiting to be built.")
        doc.workflow.steps = steps.strip()
        doc.workflow.status = "in_review"
        doc.workflow.note = ""
        doc.workflow.submitted_by = actor.email
        return "workflow.submitted", Role.WORKFLOW_BUILDER.value, actor.email, "submitted for validation"

    return store.mutate(request_id, change)


def review_workflow(store: MafStore, actor: Actor, request_id: str, approve: bool, note: str | None) -> RequestDoc:
    actor.require(Role.ETHICS_CHAMPION)
    cleaned = _note(note, approve)

    def change(tx: MafStore, doc: RequestDoc) -> tuple[str, str, str, str]:
        if doc.workflow.status != "in_review":
            raise DomainError(409, "This workflow is not awaiting review.")
        _no_self_review(actor, doc.workflow.submitted_by, "a workflow")
        doc.workflow.reviewed_by = actor.email
        if not approve:
            doc.workflow.status = "todo"
            doc.workflow.note = cleaned
            return "workflow.rejected", Role.ETHICS_CHAMPION.value, actor.email, cleaned
        doc.workflow.status = "published"
        doc.workflow.note = ""
        tx.publish_workflow(
            CatalogueWorkflow(
                name=doc.name,
                steps=doc.workflow.steps,
                agents=[agent.agent for agent in doc.agents],
                published_by=doc.workflow.submitted_by or actor.email,
                approved_by=actor.email,
                published_at=now(),
            )
        )
        return "workflow.approved", Role.ETHICS_CHAMPION.value, actor.email, "approved and published to the catalogue"

    return store.mutate(request_id, change)


def _find_agent(doc: RequestDoc, agent: str) -> AgentWork:
    for item in doc.agents:
        if name_key(item.agent) == name_key(agent):
            return item
    raise DomainError(404, "Agent is not part of this request.")


def submit_agent(store: MafStore, actor: Actor, request_id: str, agent: str, instructions: str) -> RequestDoc:
    actor.require(Role.AGENT_BUILDER)

    def change(_store: MafStore, doc: RequestDoc) -> tuple[str, str, str, str]:
        item = _find_agent(doc, agent)
        if item.status != "todo":
            raise DomainError(409, "This agent is not waiting to be built.")
        item.instructions = instructions.strip()
        item.status = "in_review"
        item.note = ""
        item.submitted_by = actor.email
        return "agent.submitted", Role.AGENT_BUILDER.value, actor.email, f"{item.agent}: submitted for certification"

    return store.mutate(request_id, change)


def review_agent(
    store: MafStore, actor: Actor, request_id: str, agent: str, approve: bool, note: str | None
) -> RequestDoc:
    actor.require(Role.GOVERNANCE_MANAGER)
    cleaned = _note(note, approve)

    def change(tx: MafStore, doc: RequestDoc) -> tuple[str, str, str, str]:
        item = _find_agent(doc, agent)
        if item.status != "in_review":
            raise DomainError(409, "This agent is not awaiting certification.")
        _no_self_review(actor, item.submitted_by, "an agent")
        item.reviewed_by = actor.email
        if not approve:
            item.status = "todo"
            item.note = cleaned
            return "agent.rejected", Role.GOVERNANCE_MANAGER.value, actor.email, f"{item.agent}: {cleaned}"
        item.status = "published"
        item.note = ""
        tx.publish_agent(
            CatalogueAgent(
                name=item.agent,
                instructions=item.instructions,
                published_by=item.submitted_by or actor.email,
                approved_by=actor.email,
                published_at=now(),
            )
        )
        return "agent.certified", Role.GOVERNANCE_MANAGER.value, actor.email, f"{item.agent}: certified and published"

    return store.mutate(request_id, change)


def check_can_evaluate(actor: Actor, doc: RequestDoc | None) -> RequestDoc:
    actor.require(Role.CHATBOT_BUILDER)
    if doc is None:
        raise DomainError(404, "Request not found.")
    if not doc.ready:
        raise DomainError(409, "The workflow and every agent must be published before evaluation.")
    if doc.released:
        raise DomainError(409, "This request has already been released.")
    return doc


def record_evaluation(store: MafStore, actor: Actor, request_id: str, evaluation: Evaluation) -> RequestDoc:
    def change(_store: MafStore, doc: RequestDoc) -> tuple[str, str, str, str]:
        if not doc.ready or doc.released:
            raise DomainError(409, "The request changed while the evaluation was running.")
        doc.evaluation = evaluation
        return (
            "evaluation.completed",
            Role.CHATBOT_BUILDER.value,
            actor.email,
            f"{evaluation.execution_id} · {evaluation.status} · {evaluation.model}",
        )

    return store.mutate(request_id, change)


def review_release(store: MafStore, actor: Actor, request_id: str, approve: bool, note: str | None) -> RequestDoc:
    actor.require(Role.ETHICS_CHAMPION)
    cleaned = _note(note, approve)

    def change(_store: MafStore, doc: RequestDoc) -> tuple[str, str, str, str]:
        if not doc.ready or doc.evaluation is None or doc.released:
            raise DomainError(409, "This request is not awaiting release review.")
        _no_self_review(actor, doc.evaluation.run_by, "an evaluation")
        if not approve:
            execution = doc.evaluation.execution_id
            doc.evaluation = None
            return "release.rejected", Role.ETHICS_CHAMPION.value, actor.email, f"{execution}: {cleaned}"
        doc.released = True
        doc.released_by = actor.email
        return "release.approved", Role.ETHICS_CHAMPION.value, actor.email, "released"

    return store.mutate(request_id, change)


def hand_over(store: MafStore, actor: Actor, request_id: str) -> RequestDoc:
    actor.require(Role.CHATBOT_BUILDER)

    def change(_store: MafStore, doc: RequestDoc) -> tuple[str, str, str, str]:
        if not doc.released or doc.handed_over:
            raise DomainError(409, "Only a released request can be handed over, once.")
        doc.handed_over = True
        return "handover.completed", Role.CHATBOT_BUILDER.value, actor.email, "handed over to the UX team"

    return store.mutate(request_id, change)
