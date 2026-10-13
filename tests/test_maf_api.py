from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from clm_aws_decision_support import webapp
from clm_aws_decision_support.auth import SupabaseUser, required_supabase_user
from clm_aws_decision_support.maf_api.router import get_runner_factory
from clm_aws_decision_support.maf_api.store import MafStore, get_maf_store


class Harness:
    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.user = "admin@example.com"

    def as_user(self, email: str) -> "Harness":
        self.user = email
        return self

    def call(self, method: str, path: str, **kwargs: Any):
        return self.client.request(method, f"/api/maf{path}", headers={"X-API-Key": "k"}, **kwargs)


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    monkeypatch.delenv("MAF_ADMIN_EMAILS", raising=False)
    monkeypatch.delenv("CLM_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("MAF_SEPARATION_OF_DUTIES", raising=False)
    store = MafStore(str(tmp_path / "maf.db"))
    holder = Harness(TestClient(webapp.app))

    async def fake_runner(question, agent_name, instructions, on_event):
        await on_event({"stage": "maf_agent", "status": "started", "agent": agent_name, "attempt": 1})
        await on_event({"stage": "maf_agent", "status": "completed", "agent": agent_name, "model": "m1"})
        await on_event({"stage": "sentinel_evaluation", "score": 0.9, "passed": True})
        return {
            "answer": "Hello from MAF " * 10,
            "model": "m1",
            "provider": "p",
            "policy": "linucb",
            "cost": 0.001,
            "elapsed_ms": 12,
            "decision_id": "d1",
            "execution_id": "exec_1",
            "sentinel": {"status": "completed", "quality": [{"score": 0.9}], "trace": []},
        }

    webapp.app.dependency_overrides[get_maf_store] = lambda: store
    webapp.app.dependency_overrides[required_supabase_user] = lambda: SupabaseUser(
        id=holder.user, email=holder.user
    )
    webapp.app.dependency_overrides[webapp.required_api_key] = lambda: webapp.ApiKeyPrincipal(
        id="key", user_id=holder.user
    )
    webapp.app.dependency_overrides[get_runner_factory] = lambda: (lambda request, key, props: fake_runner)
    yield holder
    webapp.app.dependency_overrides.clear()


def seed_team(h: Harness) -> None:
    assert h.call("GET", "/me").json()["roles"] == ["admin"]
    for email, role in [
        ("cb@example.com", "chatbot_builder"),
        ("wb@example.com", "workflow_builder"),
        ("ab@example.com", "agent_builder"),
        ("ec@example.com", "ethics_champion"),
        ("gm@example.com", "governance_manager"),
    ]:
        assert h.call("PUT", "/members", json={"email": email, "roles": [role]}).status_code == 200


def new_request(h: Harness, name: str = "Refund bot", agents: list[str] | None = None) -> dict[str, Any]:
    h.as_user("cb@example.com")
    response = h.call(
        "POST",
        "/requests",
        json={"name": name, "requirement": "Handle refunds", "agents": agents or ["Triage", "Policy"]},
    )
    assert response.status_code == 201, response.text
    return response.json()


def build_everything(h: Harness, request_id: str, agents: list[str]) -> None:
    h.as_user("wb@example.com")
    assert h.call("POST", f"/requests/{request_id}/workflow/submit", json={"steps": "a -> b"}).status_code == 200
    h.as_user("ec@example.com")
    assert h.call("POST", f"/requests/{request_id}/workflow/review", json={"approve": True}).status_code == 200
    for agent in agents:
        h.as_user("ab@example.com")
        assert (
            h.call("POST", f"/requests/{request_id}/agents/{agent}/submit", json={"instructions": f"be {agent}"}).status_code
            == 200
        )
        h.as_user("gm@example.com")
        assert h.call("POST", f"/requests/{request_id}/agents/{agent}/review", json={"approve": True}).status_code == 200


def agui_body(prompt: str = "Refund my order") -> dict[str, Any]:
    return {
        "threadId": "t",
        "runId": "r",
        "state": {},
        "messages": [{"id": "1", "role": "user", "content": prompt}],
        "tools": [],
        "context": [],
        "forwardedProps": {"policy": "linucb"},
    }


def test_full_lifecycle_and_catalogue_reuse(harness: Harness) -> None:
    h = harness
    seed_team(h)
    doc = new_request(h)
    rid = doc["id"]
    assert doc["phase"] == "building"
    assert set(doc["waiting_on"]) == {"workflow_builder", "agent_builder"}

    build_everything(h, rid, ["Triage", "Policy"])
    h.as_user("cb@example.com")
    detail = h.call("GET", f"/requests/{rid}").json()
    assert detail["phase"] == "ready_to_evaluate"

    response = h.call("POST", f"/requests/{rid}/agui", json=agui_body())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    types = [event["type"] for event in events]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    for expected in ("TOOL_CALL_START", "TOOL_CALL_RESULT", "TEXT_MESSAGE_CONTENT", "CUSTOM", "STATE_SNAPSHOT"):
        assert expected in types

    detail = h.call("GET", f"/requests/{rid}").json()
    assert detail["phase"] == "awaiting_release"
    assert detail["waiting_on"] == ["ethics_champion"]

    h.as_user("ec@example.com")
    reject = h.call("POST", f"/requests/{rid}/release/review", json={"approve": False})
    assert reject.status_code == 422
    assert h.call("POST", f"/requests/{rid}/release/review", json={"approve": False, "note": "retry"}).json()[
        "phase"
    ] == "ready_to_evaluate"

    h.as_user("cb@example.com")
    assert h.call("POST", f"/requests/{rid}/agui", json=agui_body()).status_code == 200
    h.as_user("ec@example.com")
    assert h.call("POST", f"/requests/{rid}/release/review", json={"approve": True}).json()["phase"] == "ready_for_handover"
    h.as_user("cb@example.com")
    done = h.call("POST", f"/requests/{rid}/handover").json()
    assert done["phase"] == "complete"
    assert any(event["action"] == "handover.completed" for event in done["events"])

    catalogue = h.call("GET", "/catalogue").json()
    assert [item["name"] for item in catalogue["workflows"]] == ["Refund bot"]
    assert {item["name"] for item in catalogue["agents"]} == {"Triage", "Policy"}

    again = new_request(h, name="refund  BOT", agents=["triage", "Policy"])
    assert again["workflow"]["from_catalogue"] is True
    assert again["phase"] == "ready_to_evaluate"


def test_rejection_loop_and_out_of_order_calls(harness: Harness) -> None:
    h = harness
    seed_team(h)
    rid = new_request(h, agents=["Solo"])["id"]

    h.as_user("ec@example.com")
    assert h.call("POST", f"/requests/{rid}/workflow/review", json={"approve": True}).status_code == 409
    h.as_user("cb@example.com")
    assert h.call("POST", f"/requests/{rid}/handover").status_code == 409
    assert h.call("POST", f"/requests/{rid}/agui", json=agui_body()).status_code == 409

    h.as_user("wb@example.com")
    h.call("POST", f"/requests/{rid}/workflow/submit", json={"steps": "x"})
    assert h.call("POST", f"/requests/{rid}/workflow/submit", json={"steps": "y"}).status_code == 409
    h.as_user("ec@example.com")
    rejected = h.call("POST", f"/requests/{rid}/workflow/review", json={"approve": False, "note": "add a step"}).json()
    assert rejected["workflow"]["status"] == "todo" and rejected["workflow"]["note"] == "add a step"
    assert h.call("GET", "/catalogue").json()["workflows"] == []


def test_role_enforcement_and_membership(harness: Harness) -> None:
    h = harness
    seed_team(h)
    rid = new_request(h)["id"]

    h.as_user("stranger@example.com")
    assert h.call("GET", "/requests").status_code == 403
    assert h.call("GET", "/me").json()["roles"] == []

    h.as_user("wb@example.com")
    assert h.call("POST", "/requests", json={"name": "n", "requirement": "r", "agents": ["a"]}).status_code == 403
    assert h.call("PUT", "/members", json={"email": "wb@example.com", "roles": ["admin"]}).status_code == 403
    assert (
        h.call("POST", f"/requests/{rid}/agents/Triage/submit", json={"instructions": "x"}).status_code == 403
    )

    h.as_user("admin@example.com")
    assert h.call("PUT", "/members", json={"email": "admin@example.com", "roles": []}).status_code == 409
    assert h.call("DELETE", "/members/wb@example.com").status_code == 200
    assert h.call("GET", "/requests").status_code == 200
    h.as_user("wb@example.com")
    assert h.call("GET", "/requests").status_code == 403


def test_separation_of_duties(harness: Harness, monkeypatch) -> None:
    monkeypatch.setenv("MAF_SEPARATION_OF_DUTIES", "1")
    h = harness
    seed_team(h)
    h.call("PUT", "/members", json={"email": "both@example.com", "roles": ["workflow_builder", "ethics_champion"]})
    rid = new_request(h, agents=["Solo"])["id"]
    h.as_user("both@example.com")
    h.call("POST", f"/requests/{rid}/workflow/submit", json={"steps": "x"})
    assert h.call("POST", f"/requests/{rid}/workflow/review", json={"approve": True}).status_code == 403
    h.as_user("ec@example.com")
    assert h.call("POST", f"/requests/{rid}/workflow/review", json={"approve": True}).status_code == 200


def test_agui_requires_matching_api_key_owner(harness: Harness) -> None:
    h = harness
    seed_team(h)
    rid = new_request(h, agents=["Solo"])["id"]
    build_everything(h, rid, ["Solo"])
    h.as_user("cb@example.com")
    webapp.app.dependency_overrides[webapp.required_api_key] = lambda: webapp.ApiKeyPrincipal(id="k", user_id="someone-else")
    assert h.call("POST", f"/requests/{rid}/agui", json=agui_body()).status_code == 403


def test_stream_sends_state_snapshot(harness: Harness, monkeypatch) -> None:
    from clm_aws_decision_support.maf_api import router as router_module

    h = harness
    seed_team(h)
    new_request(h)
    store = webapp.app.dependency_overrides[get_maf_store]()

    class Once:
        calls = 0

        async def is_disconnected(self) -> bool:
            Once.calls += 1
            return Once.calls > 1

    route = next(r for r in webapp.app.routes if getattr(r, "path", "") == "/api/maf/stream")
    import asyncio

    async def collect() -> str:
        response = await route.endpoint(Once(), None, store)
        out = ""
        async for chunk in response.body_iterator:
            out += chunk if isinstance(chunk, str) else chunk.decode()
        return out

    body = asyncio.run(collect())
    assert "RUN_STARTED" in body and "STATE_SNAPSHOT" in body and "Refund bot" in body
    assert router_module  # import guard
