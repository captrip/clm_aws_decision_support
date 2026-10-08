from __future__ import annotations

import os
import pathlib
import uuid
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ask import ask as run_ask
from .auth import SupabaseUser, required_supabase_user
from .cascade import Cascade, Question
from .demo import (
    SimulatedCLMClient,
    SimulatedGrader,
    SimulatedModelProvider,
    seed_demo_deployments,
    visible_answer,
)

from .features import DEFAULT_FEATURE_DIM, default_feature_fn
from .models import ModelRegistry, ModelSpec, ProviderKind
from .policies.base import InvalidSelectionError
from .policies.clm import CLMPolicy, CLMUnavailableError, default_client
from .policies.hybrid import HybridPolicy
from .policies.linucb import LinUCBPolicy, LinUCBState
from .providers import ProviderKind, ProviderRouter
from .store import Store

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DATA_DIR = pathlib.Path(os.environ.get("CLM_DATA_DIR","data")).resolve()
_DB_PATH = str(_DATA_DIR / "deployments.db")
_LINUCB_SNAPSHOT_PATH = str(_DATA_DIR / "linucb.npz")

_store = Store(_DB_PATH)
if not _store.list_diployments():
    seed_demo_deployments(_store)

_model_provider = SimulatedModelProvider(accuracy={})
_grader = SimulatedGrader()

_registry: ModelRegistry
_provider_router: ProviderRouter
_accuracy: dict[str, float]
_MODEL_DESCRIPTIONS: dict[str, str]

def _reduild_registry() -> None:
    global _registry, _provider_router, _accuracy, _MODEL_DESCRIPTIONS
    rows = _store.list_deployments()
    _registry = ModelRegistry(
        ModelSpec(
            logical_name=row["logical_name"],
            provider=ProviderKind(row["provider"]),
            model_id=row["model_id"],
            deployment=row["deployment"],
            input_cost_per_million=row["input_cost_per_million"],
            output_cost_per_million=row["output_cost_per_million"]
        )
        for row in rows
    )
    _accuracy = {
        row["logical_name"]: row["sample_accuracy"] if row["sample_accuracy"] is not None else 0.5
        for row in rows
    }
    _model_provider.accuracy = _accuracy
    _provider_router = ProviderRouter(
        _registry, {ProviderKind.BEDROCK: _model_provider, ProviderKind.AZURE_OPENAI: _model_provider}
    )
    _MODEL_DESCRIPTIONS = {name: f"{name} ({_registry[name].provider.value})" for name in _registry}

_reduild_registry()

_linucb = LinUCBState(list(_registry), dim=DEFAULT_FEATURE_DIM)
_linucb.restore(_LINUCB_SNAPSHOT_PATH)


def _clm_client():
    if os.environ.get("CLM_BASE_URL"):
        return default_client()
    return SimulatedCLMClient(quality = _accuracy)

def _build_policy(name: str):
    if name == "linucb":
        return LinUCBPolicy(_linucb), _linucb
    client = _clm_client()
    if name == "clm":
        return CLMPolicy(client=client, model_descriptions=_MODEL_DESCRIPTIONS), None
    if name == "hybrid":
        return HybridPolicy(_linucb, client=client, model_descriptions=_MODEL_DESCRIPTIONS), _linucb
    raise HTTPException(status_code=422, detail=f"unknown policy {name!r}")

class RunRequest(BaseModel):
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    budget: float | None = 0.01
    policy: Literal["linucb","clm","hybrid"] = "linucb"
    max_attempts: int = Field(default=3, ge=1, le=5)

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    budget: float | None = 0.01
    policy: Literal["linucb", "clm", "hybrid"] = "linucb"

class FeedbackRequest(BaseModel):
    decision_id: str
    rating: Literal["up","down"]

class DeploymentRequest(BaseModel):
    logical_name: str = Field(min_length=1)
    provider: Literal["bedrock", "azure_openai"]
    model_id: str | None = None
    deployment: str | None = None
    input_cost_per_million: float = Field(ge=0.0)
    output_cost_per_million: float = Field(ge=0.0)

    sample_accuracy: float = Field(default=0.5, ge=0.0, le=0.0)

class FeedbackImportRecord(BaseModel):
    question: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reward: Literal[0,1]
    cost: float = Field(default=0.0, ge=0.0)

class FeedbackImportRequest(BaseModel):
    records: list[FeedbackImportRecord] = Field(min_length=1)


class CreateApiKeyRequest(BaseModel):
    label: str | None = None

def required_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    record = _store._verify_api_key(x_api_key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return record["user_id"]

app = FastAPI(title="Routing Cascade")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def health() -> dict[str, Any]:
    return {"service": "clm-linucb-decision-support", "status": "ok"}

@app.post("/api/auth/keys")
def create_api_key(body: CreateApiKeyRequest, user: SupabaseUser = Depends(required_supabase_user)) -> dict[str, Any]:

    row, raw_key = _store.create_api_key(user.id, label = body.label)
    return {**row,"api_key": raw_key}

@app.get("/api/auth/keys")
def list_api_keys(user: SupabaseUser = Depends(required_supabase_user)) -> dict[str, Any]:
    return _store.list_api_keys(user.id)

@app.delete("/api/auth/keys/{key_id}")
def revoke_api_key(
    key_id: str,
    user: SupabaseUser = Depends(required_supabase_user),
) -> dict[str, Any]:
    if not _store.revoke_api_key(user.id, key_id):
        raise HTTPException(status_code=404, detail=f"Unknown API Key {key_id!r}")
    return {"revoked": key_id}

@app.get("/api/models")
def list_models(_user_id:str = Depends(required_api_key)) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "provider": spec.provider.value,
            "input_cost_per_million": spec.input_cost_per_million,
            "output_cost_per_million":spec.output_cost_per_million
        }
        for name, spec in _registry.items()
    ]

@app.get("/api/deployments")
def list_diployments(_user_id: str = Depends(required_api_key)) -> list[dict[str, Any]]:
    return _store.list_deployments()

@app.post("/api/deployments")
def add_deployment(
    body: DeploymentRequest,
    _user_id: str = Depends(required_api_key)
) -> dict[str,Any]:
    provider = ProviderKind(body.provider)
    try:
        ModelSpec(
            logical_name=body.logical_name,
            provider=provider,
            model_id=body.model_id,
            deployment=body.deployment,
            input_cost_per_million=body.input_cost_per_million,
            output_cost_per_million=body.output_cost_per_million
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        _store.add_deployment(
            logical_name=body.logical_name,
            provider=provider,
            model_id=body.model_id,
            deployment=body.deployment,
            input_cost_per_million=body.input_cost_per_million,
            output_cost_per_million=body.output_cost_per_million,
            sample_accuracy = body.sample_accuracy
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    _reduild_registry()
    _linucb.add_model(body.logical_name)
    _linucb.snapshot(_LINUCB_SNAPSHOT_PATH)
    return {"added": body.logical_name}

@app.delete("/api/deployments/{logical_name}")
def remove_deployment(logical_name: str, _user_id: str = Depends(required_api_key)) -> dict[str, Any]:
    if not _store.deactivate_deploymnet(logical_name):
        raise HTTPException(status_code=404, detail=f"Deployment not found {logical_name}")
    _reduild_registry()
    return {"removed": logical_name}

@app.post("/api/ask")
def ask_question(body: AskRequest, _user_id: str = Depends(required_api_key)) -> dict[str, Any]:
    try:
        policy, linucb_state = _build_policy(body.policy)
        result = run_ask(
            question=body.question,
            policy=policy,
            router=_provider_router,
            registry=_registry,
            budget=body.budget,
            linucb=linucb_state
        )
    except InvalidSelectionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except CLMUnavailableError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    decision_id = str(uuid.uuid4())
    if linucb_state is not None:
        _store.log_decision(decision_id,body.question,result.model,body.policy,result.feature, result.cost)

    return{
        "decision_id": decision_id,
        "question": body.question,
        "policy": body.policy,
        "model": result.model,
        "provider": result.provider,
        "answer": visible_answer(result.answer),
        "cost": result.cost,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens
        },
        "rationale": result.selection.rationale,
        "candidates": {model: dict(scores) for model, scores in result.selection.scores.items()},
        "retried_models": list(result.retried_models),
        "feedback_eligible": linucb_state is not None,
    }

@app.post("/api/feedback")
def submit_feedbck(body: FeedbackRequest, _user_id: str = Depends(required_api_key)) -> dict[str,Any]:
    reward = 1.0 if body.rating == "up" else 0.0
    decision = _store.attach_reward(body.decision_id, reward)
    if decision is None:
        return {"applied": False, "reason": "unknown or already rated"}

    _linucb.update(model=decision["model"], features=decision["features"], reward=reward,cost=decision["cost"])
    _linucb.snapshot(_LINUCB_SNAPSHOT_PATH)
    return {"applied": True}

@app.post("/api/feedback/import")
def import_feedback(body: FeedbackImportRequest, _user_id: str = Depends(required_api_key)) -> dict[str, Any]:
    applied = 0
    skipped: list[dict[str, Any]] =[]
    for i, record in enumerate(body.records):
        if record.model not in _registry:
            skipped.append({"index":i,"reason": f"unknown model {record.model!r}"})
            continue
        features = default_feature_fn(record.question, 0.0)
        _linucb.update(model=record.model, features=features, reward= float(record.reward), cost= record.cost)
        _store.log_decision(
            str(uuid.uuid4()),
            record.question,
            record.model,
            "import",
            features, 
            record.cost,
            reward = float(record.reward),
            source = "import"
        )
        applied += 1

    if applied:
        _linucb.snapshot(_LINUCB_SNAPSHOT_PATH)
    return {"applied": applied, "skipped": skipped}

@app.post("/api/run")
def run_cascade(body: RunRequest, _user_id: str = Depends(required_api_key)) -> dict[str, Any]:
    try:
        policy, linucb_state = _build_policy(body.policy)
        cascade = Cascade(
            policy=policy,
            router=_provider_router,
            grader=_grader,
            registry=_registry,
            max_attempts=body.max_attempts,
            linucb=linucb_state
        )
        result = cascade.run(
            question=Question(body.question),
            expected_answer= body.expected_answer,
            budget=body.budget
        )

    except InvalidSelectionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except CLMUnavailableError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    if linucb_state is not None:
        _linucb.snapshot(_LINUCB_SNAPSHOT_PATH)

    return {
        "policy": body.policy,
        "question": body.question,
        "expected_answer": body.expected_answer,
        "budget": body.budget,
        "success": result.success,
        "total_cost": result.total_cost,
        "attempts":[
            {
                "step": attempt.step,
                "model": attempt.model,
                "provider": attempt.provider,
                "policy": attempt.policy,
                "cost": attempt.cost,
                "usage": {
                    "input_tokens": attempt.usage.input_tokens,
                    "output_tokens": attempt.usage.output_tokens
                },
                "reward": attempt.reward,
                "error": attempt.error,
                "rationale": attempt.selection.rationale,
                "candidates": {model: dict(scores) for model, scores in attempt.selection.scores.items()},

            }
            for attempt in result.attempts
        ],
    }