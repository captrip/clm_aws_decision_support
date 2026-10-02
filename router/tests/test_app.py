import numpy as np
import pytest
from fastapi.testclient import TestClient

from router import config
from router.app import acquire_single_worker_lock, build_app


class StubRouter:
    """Stands in for policy.Router at the HTTP boundary."""

    def __init__(self):
        self.calls = []
        self.rewards = []
        self.result = {
            "text": "the answer",
            "arm": config.ARM_NAMES[0],
            "cascade_depth": 1,
            "cost": 0.00012,
            "decision_ids": [7],
            "verdict": 1,
            "status": "success",
            "attempts": [],
        }
        self.bandit = type(
            "B", (), {"models": {a: {"pulls": 0} for a in config.ARM_NAMES}}
        )()

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.result

    def record_reward(self, decision_id, reward):
        self.rewards.append((decision_id, reward))
        return True


@pytest.fixture
def client():
    stub = StubRouter()
    app = build_app(router=stub, lock=False)
    with TestClient(app) as test_client:
        test_client.stub = stub
        yield test_client


def test_healthz_reports_the_arms(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["arms"]) == 7


def test_complete_returns_the_routed_answer(client):
    response = client.post(
        "/v1/complete",
        json={"conversation_id": "c1", "query": "what is 2+2", "task_type": "math"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "the answer"
    assert body["arm"] == config.ARM_NAMES[0]
    assert body["cascade_depth"] == 1
    assert body["cost"] == pytest.approx(0.00012)
    assert body["decision_ids"] == [7]
    assert body["status"] == "success"


def test_complete_passes_fields_through_to_the_router(client):
    client.post(
        "/v1/complete",
        json={
            "conversation_id": "c2",
            "query": "solve it",
            "task_type": "math",
            "expected": "42",
            "turn_index": 3,
            "conversation_tokens": 900,
        },
    )
    call = client.stub.calls[0]
    assert call["conversation_id"] == "c2"
    assert call["query"] == "solve it"
    assert call["task_type"] == "math"
    assert call["expected"] == "42"
    assert call["turn_index"] == 3
    assert call["conversation_tokens"] == 900


def test_task_type_defaults_to_prose(client):
    client.post("/v1/complete", json={"conversation_id": "c", "query": "hi"})
    assert client.stub.calls[0]["task_type"] == "prose"


def test_an_invalid_task_type_is_rejected(client):
    response = client.post(
        "/v1/complete",
        json={"conversation_id": "c", "query": "hi", "task_type": "poetry"},
    )
    assert response.status_code == 422


def test_a_missing_query_is_rejected(client):
    response = client.post("/v1/complete", json={"conversation_id": "c"})
    assert response.status_code == 422


def test_an_empty_query_is_rejected(client):
    response = client.post(
        "/v1/complete", json={"conversation_id": "c", "query": "   "}
    )
    assert response.status_code == 422


def test_the_serving_arm_is_reported_in_a_header(client):
    """Routing is non-deterministic, so callers need to see what served."""
    response = client.post(
        "/v1/complete", json={"conversation_id": "c", "query": "hi"}
    )
    assert response.headers["x-router-arm"] == config.ARM_NAMES[0]
    assert response.headers["x-router-cascade-depth"] == "1"


def test_reward_endpoint_forwards_to_the_router(client):
    response = client.post("/v1/reward", json={"decision_id": 7, "reward": 1})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "applied": True}
    assert client.stub.rewards == [(7, 1)]


def test_a_duplicate_reward_reports_applied_false(client):
    """record_reward is idempotent; callers see which call actually landed."""
    client.stub.record_reward = lambda decision_id, reward: False
    response = client.post("/v1/reward", json={"decision_id": 7, "reward": 1})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "applied": False}


def test_reward_outside_zero_or_one_is_rejected(client):
    response = client.post("/v1/reward", json={"decision_id": 7, "reward": 5})
    assert response.status_code == 422


def test_reward_for_an_unknown_decision_returns_404(client):
    def boom(decision_id, reward):
        raise KeyError("nope")

    client.stub.record_reward = boom
    response = client.post("/v1/reward", json={"decision_id": 999, "reward": 1})
    assert response.status_code == 404


def test_there_is_no_streaming_endpoint(client):
    """Streaming and silent escalation are incompatible (DESIGN.md section 5).

    Scoped to the /v1 API surface: /viz/stream is an observer event stream, not
    model output, so it cannot expose a retracted answer to a caller.
    """
    paths = set(build_app(router=StubRouter(), lock=False).openapi()["paths"])
    api_paths = {p for p in paths if p.startswith("/v1/")}
    assert api_paths, "guard against the filter silently matching nothing"
    assert not any("stream" in path for path in api_paths)


class FakeEmbedder:
    def encode(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return rng.normal(size=config.EMBED_DIM)


def test_admin_judge_drains_the_queue_and_returns_its_stats(tmp_path, monkeypatch):
    """judge.run_judge_batch was complete but unreachable, so `prose` -- the
    default task type and the only one that never verifies -- got no reward
    signal at all. Operator-triggered by design; a scheduler is out of scope."""
    from router.bandit import LinUCB
    from router.context import ContextBuilder
    from router.cost import CostEstimator
    from router.policy import Router
    from router.store import Store

    def converse(client, model_id, prompt, **kwargs):
        # Stands in for both an arm call and the judge call. No AWS.
        return ("True" if model_id == config.JUDGE_MODEL else "prose answer"), {
            "inputTokens": 10,
            "outputTokens": 5,
        }

    store = Store(str(tmp_path / "judge.db"))
    real_router = Router(
        bandit=LinUCB(arm_names=config.ARM_NAMES),
        context_builder=ContextBuilder(embedder=FakeEmbedder()),
        cost_estimator=CostEstimator(),
        store=store,
        client=object(),
        converse_fn=converse,
        verify_fn=lambda *a, **k: None,
    )
    monkeypatch.setattr(config, "JUDGE_SAMPLE_RATE", 1.0)
    # The lifespan snapshots a real router on shutdown; keep it out of
    # router/state/.
    monkeypatch.setattr(config, "SNAPSHOT_PATH", str(tmp_path / "bandit.npz"))
    monkeypatch.setattr(config, "COST_STATE_PATH", str(tmp_path / "cost.npz"))

    result = real_router.complete("conv", "write me a poem", task_type="prose")
    decision_id = result["decision_ids"][0]
    assert store.get_decision(decision_id)["reward"] is None

    app = build_app(router=real_router, lock=False)
    with TestClient(app) as test_client:
        response = test_client.post("/admin/judge")
    assert response.status_code == 200
    assert set(response.json()) == {
        "considered",
        "judged",
        "skipped",
        "ambiguous",
        "errors",
    }
    assert response.json()["judged"] == 1
    assert store.get_decision(decision_id)["reward"] == 1
    store.close()


def test_a_cold_start_is_logged_loudly_enough_to_reach_an_operator(caplog):
    """uvicorn leaves the root logger at WARNING with no handlers, so an INFO
    line would go nowhere under the documented launch command. A cold start
    means every pull of learning is gone; it has to be visible."""
    import logging

    from router.app import log_restore

    with caplog.at_level(logging.INFO, logger="router.app"):
        log_restore(False, "/nope/bandit.npz", "snapshot", lambda: "unused")
        log_restore(True, "/yes/bandit.npz", "snapshot", lambda: "12 pulls")

    cold, warm = caplog.records
    assert cold.levelno >= logging.WARNING
    assert "starting cold" in cold.getMessage()
    # The warm case is routine, so INFO is right for it.
    assert warm.levelno == logging.INFO
    assert "12 pulls" in warm.getMessage()


def test_the_lock_blocks_a_second_holder(tmp_path):
    path = str(tmp_path / "router.lock")
    first = acquire_single_worker_lock(path)
    try:
        with pytest.raises(RuntimeError, match="single worker"):
            acquire_single_worker_lock(path)
    finally:
        first.close()


def test_the_lock_can_be_reacquired_after_release(tmp_path):
    path = str(tmp_path / "router.lock")
    first = acquire_single_worker_lock(path)
    first.close()
    second = acquire_single_worker_lock(path)
    second.close()


def test_the_default_lock_blocks_a_second_holder_from_another_cwd(
    tmp_path, monkeypatch
):
    """A cwd-relative LOCK_PATH resolves to a different file per working
    directory, so both workers take a lock and each keeps its own bandit state --
    the exact invariant DESIGN.md section 3 says must be impossible."""
    first = acquire_single_worker_lock()
    try:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(RuntimeError, match="single worker"):
            acquire_single_worker_lock()
    finally:
        first.close()
