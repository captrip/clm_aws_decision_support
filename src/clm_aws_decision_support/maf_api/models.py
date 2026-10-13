"""Domain models for the MAF workflow and agent lifecycle."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    ADMIN = "admin"
    CHATBOT_BUILDER = "chatbot_builder"
    WORKFLOW_BUILDER = "workflow_builder"
    AGENT_BUILDER = "agent_builder"
    ETHICS_CHAMPION = "ethics_champion"
    GOVERNANCE_MANAGER = "governance_manager"


Status = Literal["todo", "in_review", "published"]


def name_key(name: str) -> str:
    return " ".join(name.lower().split())


class Evaluation(BaseModel):
    execution_id: str
    status: str
    model: str
    provider: str | None = None
    policy: str | None = None
    answer: str
    prompt: str
    cost: float = 0.0
    elapsed_ms: float = 0.0
    decision_id: str | None = None
    quality: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    run_by: str
    run_at: str


class WorkflowWork(BaseModel):
    status: Status = "todo"
    from_catalogue: bool = False
    steps: str = ""
    note: str = ""
    submitted_by: str | None = None
    reviewed_by: str | None = None


class AgentWork(BaseModel):
    agent: str
    status: Status = "todo"
    from_catalogue: bool = False
    instructions: str = ""
    note: str = ""
    submitted_by: str | None = None
    reviewed_by: str | None = None


class RequestDoc(BaseModel):
    id: str
    name: str
    requirement: str
    created_by: str
    created_at: str
    updated_at: str
    workflow: WorkflowWork
    agents: list[AgentWork]
    evaluation: Evaluation | None = None
    released: bool = False
    released_by: str | None = None
    handed_over: bool = False

    @property
    def workflow_ready(self) -> bool:
        return self.workflow.status == "published"

    @property
    def agents_ready(self) -> bool:
        return all(agent.status == "published" for agent in self.agents)

    @property
    def ready(self) -> bool:
        return self.workflow_ready and self.agents_ready

    @property
    def phase(self) -> str:
        if not self.ready:
            return "building"
        if self.evaluation is None:
            return "ready_to_evaluate"
        if not self.released:
            return "awaiting_release"
        if not self.handed_over:
            return "ready_for_handover"
        return "complete"

    @property
    def waiting_on(self) -> list[Role]:
        waiting: set[Role] = set()
        if self.workflow.status == "todo":
            waiting.add(Role.WORKFLOW_BUILDER)
        elif self.workflow.status == "in_review":
            waiting.add(Role.ETHICS_CHAMPION)
        for agent in self.agents:
            if agent.status == "todo":
                waiting.add(Role.AGENT_BUILDER)
            elif agent.status == "in_review":
                waiting.add(Role.GOVERNANCE_MANAGER)
        if self.ready:
            if self.evaluation is None or (self.released and not self.handed_over):
                waiting.add(Role.CHATBOT_BUILDER)
            elif not self.released:
                waiting.add(Role.ETHICS_CHAMPION)
        return sorted(waiting, key=lambda role: role.value)

    def public(self, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["phase"] = self.phase
        data["waiting_on"] = [role.value for role in self.waiting_on]
        if events is not None:
            data["events"] = events
        return data


class CatalogueWorkflow(BaseModel):
    name: str
    steps: str
    agents: list[str]
    published_by: str
    approved_by: str
    published_at: str
    version: int = 1


class CatalogueAgent(BaseModel):
    name: str
    instructions: str
    published_by: str
    approved_by: str
    published_at: str
    version: int = 1


class CreateRequestBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    requirement: str = Field(min_length=1, max_length=1000)
    agents: list[str] = Field(min_length=1, max_length=12)

    @field_validator("name", "requirement")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("agents")
    @classmethod
    def _agents(cls, value: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        for raw in value:
            cleaned = " ".join(raw.split())
            if not cleaned or len(cleaned) > 80:
                raise ValueError("agent names must be 1-80 characters")
            seen.setdefault(name_key(cleaned), cleaned)
        return list(seen.values())


class SubmitWorkflowBody(BaseModel):
    steps: str = Field(min_length=1, max_length=2000)


class SubmitAgentBody(BaseModel):
    instructions: str = Field(min_length=1, max_length=2000)


class ReviewBody(BaseModel):
    approve: bool
    note: str | None = Field(default=None, max_length=500)


class MemberBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    roles: list[Role] = Field(max_length=len(Role))

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value:
            raise ValueError("must be an email address")
        return value


class EvaluationProps(BaseModel):
    policy: Literal["linucb", "clm", "hybrid"] = "linucb"
    quality_threshold: float = Field(default=0.65, ge=0, le=1)
    budget: float | None = Field(default=0.01, ge=0)
