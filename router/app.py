"""HTTP surface. One buffered endpoint, single worker only.

DESIGN.md section 3: two workers would each hold their own A/b and snapshot to
the same file, so the last writer silently discards the other's learning. The
service keeps serving and quietly stops learning, which is why this is enforced
with an exclusive file lock at boot rather than left to documentation.
"""

import fcntl
import logging
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from router import config, judge, viz

_log = logging.getLogger(__name__)
_STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"


def acquire_single_worker_lock(path=config.LOCK_PATH):
    """Hold an exclusive lock for the process lifetime, or refuse to boot."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(path, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(
            f"another process already holds {path}. This service must run as a "
            f"single worker (uvicorn --workers 1): each worker would keep its "
            f"own bandit state and overwrite the others' snapshot."
        )
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


class CompleteRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    task_type: str = "prose"
    expected: str | None = None
    schema_: dict | None = Field(default=None, alias="schema")
    turn_index: int = 0
    conversation_tokens: int = 0
    budget: float | None = None

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value):
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("task_type")
    @classmethod
    def task_type_known(cls, value):
        if value not in config.TASK_TYPES:
            raise ValueError(f"task_type must be one of {config.TASK_TYPES}")
        return value


class RewardRequest(BaseModel):
    decision_id: int
    reward: int

    @field_validator("reward")
    @classmethod
    def reward_is_binary(cls, value):
        if value not in (0, 1):
            raise ValueError("reward must be 0 or 1")
        return value


def log_restore(restored, path, label, describe):
    """Report whether persisted state came back, and how much.

    A cold start discards every pull of learning, so it must never be something
    an operator has to infer from behaviour -- which is why the cold branch logs
    at WARNING, not INFO. uvicorn's logging config defines only its own loggers
    and leaves root at WARNING with no handlers, so an INFO line here would go
    nowhere under the launch command in README.md; WARNING still reaches stderr
    through logging.lastResort.
    """
    if restored:
        _log.info("restored %s from %s", describe(), path)
    else:
        _log.warning("no %s at %s, starting cold", label, path)


def build_default_router():
    from router import arms
    from router.bandit import LinUCB
    from router.context import ContextBuilder
    from router.cost import CostEstimator
    from router.policy import Router
    from router.store import Store

    config.assert_fallbacks_fit_budget()

    bandit = LinUCB(arm_names=config.ARM_NAMES)
    log_restore(
        bandit.restore(config.SNAPSHOT_PATH),
        config.SNAPSHOT_PATH,
        "snapshot",
        lambda: "%d pulls" % sum(m["pulls"] for m in bandit.models.values()),
    )

    cost_estimator = CostEstimator()
    log_restore(
        cost_estimator.restore(config.COST_STATE_PATH),
        config.COST_STATE_PATH,
        "cost state",
        lambda: "%d cost samples"
        % sum(s["overall"]["count"] for s in cost_estimator.stats.values()),
    )

    observer = viz.VizObserver()
    return (
        Router(
            bandit=bandit,
            context_builder=ContextBuilder(),
            cost_estimator=cost_estimator,
            store=Store(config.DB_PATH),
            client=arms.make_client(),
            observer=observer,
        ),
        observer,
    )


def build_app(router=None, lock=True):
    @asynccontextmanager
    async def lifespan(app):
        yield
        # Persist on shutdown. on_event is deprecated in current FastAPI.
        router_obj = app.state.router
        if hasattr(router_obj, "bandit") and hasattr(router_obj.bandit, "snapshot"):
            router_obj.bandit.snapshot(config.SNAPSHOT_PATH)
        if hasattr(router_obj, "cost_estimator"):
            router_obj.cost_estimator.snapshot(config.COST_STATE_PATH)

    app = FastAPI(title="Per-turn LLM router", lifespan=lifespan)
    app.state.lock_handle = acquire_single_worker_lock() if lock else None
    if router is not None:
        app.state.router = router
        app.state.viz = getattr(router, "observer", None) or viz.VizObserver()
    else:
        app.state.router, app.state.viz = build_default_router()

    @app.get("/healthz")
    def healthz():
        bandit = app.state.router.bandit
        return {
            "status": "ok",
            "arms": config.ARM_NAMES,
            "pulls": {
                arm: bandit.models[arm]["pulls"] for arm in bandit.models
            },
        }

    @app.post("/v1/complete")
    def complete(request: CompleteRequest, response: Response):
        result = app.state.router.complete(
            conversation_id=request.conversation_id,
            query=request.query,
            turn_index=request.turn_index,
            task_type=request.task_type,
            expected=request.expected,
            schema=request.schema_,
            conversation_tokens=request.conversation_tokens,
            budget=request.budget,
        )
        response.headers["x-router-arm"] = str(result["arm"])
        response.headers["x-router-cascade-depth"] = str(result["cascade_depth"])
        return {
            "text": result["text"],
            "arm": result["arm"],
            "cascade_depth": result["cascade_depth"],
            "cost": result["cost"],
            "status": result["status"],
            "decision_ids": result["decision_ids"],
        }

    @app.post("/admin/judge")
    def admin_judge():
        """Run one sampled-judge batch and return its stats verbatim.

        Operator-triggered by design: a scheduler is out of scope. Without this
        the judge is unreachable, so `prose` -- the default task type and the
        only one that never verifies -- receives no reward signal at all
        (DESIGN.md section 6 wants both sources feeding the one update path).
        """
        router_obj = app.state.router
        return judge.run_judge_batch(
            router_obj,
            client=router_obj.client,
            # Read at call time, not bound as a default, so config stays the one
            # place the rate lives. converse_fn comes from the router so a
            # test's injected stub reaches the judge too.
            sample_rate=config.JUDGE_SAMPLE_RATE,
            converse_fn=router_obj.converse_fn,
        )

    @app.post("/v1/reward")
    def reward(request: RewardRequest):
        try:
            applied = app.state.router.record_reward(
                request.decision_id, request.reward
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown decision_id")
        # applied=False means the decision already had a reward; the call is a
        # no-op, not an error, so a retrying caller can stop retrying.
        return {"ok": True, "applied": bool(applied)}

    @app.get("/viz/")
    def viz_index():
        """The visualization page. Operator surface, no auth -- same as
        /admin/judge (DESIGN.md section 9)."""
        return FileResponse(_STATIC_DIR / "viz.html", media_type="text/html")

    @app.get("/viz/events")
    def viz_events(since: int = 0):
        observer = app.state.viz
        return {"events": observer.recent(since), "stats": observer.stats()}

    @app.get("/viz/stats")
    def viz_stats():
        """Just the counters. The page polls this every few seconds, and
        reading them via /viz/events?since=0 meant downloading the whole ring
        -- 617 KB measured -- to use 48 bytes of it, while holding the
        observer's lock and a threadpool thread Router.complete shares."""
        return app.state.viz.stats()

    @app.get("/viz/stream")
    def viz_stream(since: int = 0):
        return StreamingResponse(
            viz.sse_stream(app.state.viz, since=since),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
