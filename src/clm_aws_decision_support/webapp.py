from __future__ import annotations

import io
import os
import pathlib
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .ask import ask as run_ask
from .auth import SupabaseUser, required_supabase_user
from .azure_openai import AzureOpenAIProvider
from .bedrock import BedrockProvider
from .cascade import Cascade, Question
from .demo import (
    SimulatedCLMClient,
    SimulatedGrader,
    SimulatedModelProvider,
    seed_demo_deployments,
    visible_answer,
)

from .features import DEFAULT_FEATURE_DIM, default_feature_fn
from .grader import AnswerGrader
from .models import GraderSpec, ModelRegistry, ModelSpec, ProviderKind
from .policies.base import InvalidSelectionError
from .policies.clm import CLMPolicy, CLMUnavailableError, default_client
from .policies.hybrid import HybridPolicy
from .policies.linucb import LinUCBPolicy, LinUCBState
from .providers import ProviderRouter
from .store import Store

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DATA_DIR = pathlib.Path(os.environ.get("CLM_DATA_DIR","data")).resolve()
_DB_PATH = str(_DATA_DIR / "deployments.db")
_LINUCB_SNAPSHOT_PATH = str(_DATA_DIR / "linucb.npz")
_RUNTIME_MODE = os.environ.get("CLM_RUNTIME_MODE", "demo").strip().lower()
if _RUNTIME_MODE not in {"demo", "production"}:
    raise RuntimeError("CLM_RUNTIME_MODE must be either 'demo' or 'production'.")

_store = Store(_DB_PATH)

def _initialize_deployments(store: Store, mode: str) -> None:
    if mode == "production" and store.get_metadata("registry_source") == "demo":
        raise RuntimeError(
            "Production mode cannot use a demo-seeded deployment database. "
            "Configure a clean production CLM_DATA_DIR and add reviewed deployments."
        )
    if mode == "demo" and not store.list_deployments():
        seed_demo_deployments(store)
        store.set_metadata("registry_source", "demo")


_initialize_deployments(_store, _RUNTIME_MODE)

_demo_model_provider = SimulatedModelProvider(accuracy={})
_grader = SimulatedGrader()

_registry: ModelRegistry
_provider_router: ProviderRouter
_provider_adapters: dict[ProviderKind, Any]
_accuracy: dict[str, float]
_MODEL_DESCRIPTIONS: dict[str, str]

def _provider_adapters_for_mode(mode: str) -> dict[ProviderKind, Any]:
    if mode == "demo":
        return {
            ProviderKind.BEDROCK: _demo_model_provider,
            ProviderKind.AZURE_OPENAI: _demo_model_provider,
        }
    if mode == "production":
        return {
            ProviderKind.BEDROCK: BedrockProvider(),
            ProviderKind.AZURE_OPENAI: AzureOpenAIProvider(),
        }
    raise ValueError(f"unknown runtime mode {mode!r}")

def _reduild_registry() -> None:
    global _registry, _provider_router, _provider_adapters, _accuracy, _MODEL_DESCRIPTIONS
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
    if _RUNTIME_MODE == "demo":
        _demo_model_provider.accuracy = _accuracy
    _provider_adapters = _provider_adapters_for_mode(_RUNTIME_MODE)
    _provider_router = ProviderRouter(_registry, _provider_adapters)
    _MODEL_DESCRIPTIONS = {name: f"{name} ({_registry[name].provider.value})" for name in _registry}

_reduild_registry()

_linucb = LinUCBState(list(_registry), dim=DEFAULT_FEATURE_DIM)
_linucb.restore(_LINUCB_SNAPSHOT_PATH)


def _clm_client():
    if os.environ.get("CLM_BASE_URL"):
        return default_client()
    if _RUNTIME_MODE == "production":
        raise CLMUnavailableError("CLM_BASE_URL must be configured for clm and hybrid policies.")
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

def _build_grader():
    if _RUNTIME_MODE == "demo":
        return _grader
    provider_name = os.environ.get("CLM_GRADER_PROVIDER")
    model_name = os.environ.get("CLM_GRADER_MODEL")
    if not provider_name or not model_name:
        raise HTTPException(
            status_code=503,
            detail="CLM_GRADER_PROVIDER and CLM_GRADER_MODEL are required for production evaluations.",
        )
    try:
        provider_kind = ProviderKind(provider_name)
    except ValueError as error:
        raise HTTPException(
            status_code=503,
            detail="CLM_GRADER_PROVIDER must be bedrock or azure_openai.",
        ) from error
    return AnswerGrader(
        provider=_provider_adapters[provider_kind],
        grader=GraderSpec(provider=provider_kind, model_id=model_name),
    )

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


class MAFDemoMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class MAFDemoRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    agent_name: str = Field(default="Workspace Assistant", min_length=1, max_length=80)
    instructions: str = Field(
        default="Be a reliable conversational assistant. Be direct, accurate, and clear. "
        "If you are unsure, say so.",
        min_length=1,
        max_length=2000,
    )
    conversation: list[MAFDemoMessage] = Field(default_factory=list, max_length=10)
    budget: float | None = Field(default=0.01, ge=0)
    policy: Literal["linucb", "clm", "hybrid"] = "linucb"
    quality_threshold: float = Field(default=0.65, ge=0, le=1)


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

    sample_accuracy: float = Field(default=0.5, ge=0.0, le=1.0)

class FeedbackImportRecord(BaseModel):
    question: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reward: Literal[0,1]
    cost: float = Field(default=0.0, ge=0.0)

class FeedbackImportRequest(BaseModel):
    records: list[FeedbackImportRecord] = Field(min_length=1)


class CreateApiKeyRequest(BaseModel):
    label: str | None = None

@dataclass(frozen=True, slots=True)
class ApiKeyPrincipal:
    id: str
    user_id: str


def required_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> ApiKeyPrincipal:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    record = _store.verify_api_key(x_api_key)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return ApiKeyPrincipal(id=record["id"], user_id=record["user_id"])

app = FastAPI(
    title="MONA API",
    version="1.0.0",
    description="API-key-authenticated, policy-driven model selection and invocation.",
)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CLM_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

from .maf_api import create_maf_router  # noqa: E402

app.include_router(create_maf_router(api_key_dependency=required_api_key))

_PRODUCT_SOURCES = {
    "sentinel": ("products/sentinel", "sentinel"),
    "maf": ("products/maf", "maf"),
}
_ARCHIVE_EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


@app.get("/")
def health() -> dict[str, Any]:
    return {"service": "clm-linucb-decision-support", "status": "ok"}


@app.post("/api/maf/demo")
async def run_maf_conversation_demo(
    body: MAFDemoRequest,
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    _principal: ApiKeyPrincipal = Depends(required_api_key),
) -> dict[str, Any]:
    try:
        from integration.conversational_demo import run_conversational_demo
        from sentinal import MonaAPIError, MonaClient
    except ImportError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The MAF/Sentinel demo runtime is not installed. Install the "
                "products/maf and products/sentinel packages."
            ),
        ) from error

    api_url = os.environ.get("MONA_API_URL", str(request.base_url).rstrip("/"))
    client = MonaClient(api_key=x_api_key, base_url=api_url)
    try:
        return await run_conversational_demo(
            client=client,
            question=body.question,
            agent_name=body.agent_name.strip(),
            instructions=body.instructions.strip(),
            conversation=[message.model_dump() for message in body.conversation],
            policy=body.policy,
            budget=body.budget,
            quality_threshold=body.quality_threshold,
        )
    except MonaAPIError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.get("/api/products/{product}/download")
def download_product(
    product: str,
    _principal: ApiKeyPrincipal = Depends(required_api_key),
) -> StreamingResponse:
    source_info = _PRODUCT_SOURCES.get(product)
    if source_info is None:
        raise HTTPException(status_code=404, detail="Unknown product package")

    source_path = pathlib.Path(__file__).resolve().parents[2] / source_info[0]
    if not source_path.is_dir():
        raise HTTPException(status_code=404, detail="Product package is unavailable")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source_path.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative_path = path.relative_to(source_path)
            if any(part in _ARCHIVE_EXCLUDED_DIRS for part in relative_path.parts):
                continue
            if (
                path.name == ".env"
                or path.name.startswith(".env.")
                or path.suffix in {
                    ".db",
                    ".log",
                    ".npz",
                    ".pyc",
                    ".sqlite",
                    ".sqlite3",
                }
            ):
                continue
            bundle.write(path, pathlib.Path(source_info[1]) / relative_path)

    archive.seek(0)
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{source_info[1]}-package.zip"'
            )
        },
    )


@app.post("/api/auth/keys")
def create_api_key(body: CreateApiKeyRequest, user: SupabaseUser = Depends(required_supabase_user)) -> dict[str, Any]:

    row, raw_key = _store.create_api_key(user.id, label = body.label)
    return {**row,"api_key": raw_key}

@app.get("/api/auth/keys")
def list_api_keys(user: SupabaseUser = Depends(required_supabase_user)) -> list[dict[str, Any]]:
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
def list_models(_principal: ApiKeyPrincipal = Depends(required_api_key)) -> list[dict[str, Any]]:
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
def list_diployments(_principal: ApiKeyPrincipal = Depends(required_api_key)) -> list[dict[str, Any]]:
    return _store.list_deployments()

@app.post("/api/deployments")
def add_deployment(
    body: DeploymentRequest,
    _principal: ApiKeyPrincipal = Depends(required_api_key)
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
def remove_deployment(logical_name: str, _principal: ApiKeyPrincipal = Depends(required_api_key)) -> dict[str, Any]:
    if not _store.deactivate_deployment(logical_name):
        raise HTTPException(status_code=404, detail=f"Deployment not found {logical_name}")
    _reduild_registry()
    return {"removed": logical_name}

@app.post("/api/ask")
def ask_question(body: AskRequest, principal: ApiKeyPrincipal = Depends(required_api_key)) -> dict[str, Any]:
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
    _store.record_api_usage(
        api_key_id=principal.id,
        user_id=principal.user_id,
        decision_id=decision_id,
        model=result.model,
        policy=body.policy,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost=result.cost,
    )
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

@app.get("/api/usage")
def get_api_usage(principal: ApiKeyPrincipal = Depends(required_api_key)) -> dict[str, Any]:
    return _store.get_api_usage(principal.user_id)

@app.post("/api/feedback")
def submit_feedbck(body: FeedbackRequest, _principal: ApiKeyPrincipal = Depends(required_api_key)) -> dict[str,Any]:
    reward = 1.0 if body.rating == "up" else 0.0
    decision = _store.attach_reward(body.decision_id, reward)
    if decision is None:
        return {"applied": False, "reason": "unknown or already rated"}

    _linucb.update(model=decision["model"], features=decision["features"], reward=reward,cost=decision["cost"])
    _linucb.snapshot(_LINUCB_SNAPSHOT_PATH)
    return {"applied": True}

@app.post("/api/feedback/import")
def import_feedback(body: FeedbackImportRequest, _principal: ApiKeyPrincipal = Depends(required_api_key)) -> dict[str, Any]:
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
def run_cascade(body: RunRequest, _principal: ApiKeyPrincipal = Depends(required_api_key)) -> dict[str, Any]:
    try:
        policy, linucb_state = _build_policy(body.policy)
        cascade = Cascade(
            policy=policy,
            router=_provider_router,
            grader=_build_grader(),
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