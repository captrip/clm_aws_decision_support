from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from sentinal.integrations.mona import MonaAPIError, MonaClient


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.body = json.dumps(payload).encode()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_client_sends_api_key_and_policy_routing_payload() -> None:
    client = MonaClient(api_key="wb_test_key", base_url="https://mona.example/")
    response_data = {"model": "fast", "policy": "linucb", "answer": "Routed"}

    with patch(
        "sentinal.integrations.mona.urlopen",
        return_value=FakeResponse(response_data),
    ) as open_url:
        result = client.ask("Summarize this", policy="linucb", budget=0.05)

    request = open_url.call_args.args[0]
    assert request.full_url == "https://mona.example/api/ask"
    assert request.get_header("X-api-key") == "wb_test_key"
    assert json.loads(request.data) == {
        "question": "Summarize this",
        "policy": "linucb",
        "budget": 0.05,
    }
    assert result == response_data


def test_client_reads_key_and_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONA_API_KEY", "wb_environment_key")
    monkeypatch.setenv("MONA_API_URL", "https://mona.example")
    client = MonaClient()

    with patch(
        "sentinal.integrations.mona.urlopen",
        return_value=FakeResponse({"answer": "ok"}),
    ) as open_url:
        client.ask("Question")

    assert open_url.call_args.args[0].get_header("X-api-key") == "wb_environment_key"


def test_client_requires_credentials_and_valid_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONA_API_KEY", raising=False)
    monkeypatch.delenv("MONA_API_URL", raising=False)
    with pytest.raises(ValueError, match="MONA_API_KEY"):
        MonaClient()

    client = MonaClient(api_key="key", base_url="https://mona.example")
    with pytest.raises(ValueError, match="question"):
        client.ask("  ")
    with pytest.raises(ValueError, match="policy"):
        client.ask("Question", policy="guess")  # type: ignore[arg-type]


def test_client_surfaces_backend_errors_without_hiding_http_status() -> None:
    client = MonaClient(api_key="key", base_url="https://mona.example")
    error = HTTPError(
        "https://mona.example/api/ask",
        401,
        "Unauthorized",
        {},
        io.BytesIO(b'{"detail":"Invalid API Key"}'),
    )
    with patch("sentinal.integrations.mona.urlopen", side_effect=error):
        with pytest.raises(MonaAPIError, match="HTTP 401"):
            client.ask("Question")


def test_async_client_runs_sync_request_in_worker_thread() -> None:
    client = MonaClient(api_key="key", base_url="https://mona.example")
    with patch(
        "sentinal.integrations.mona.urlopen",
        return_value=FakeResponse({"answer": "async"}),
    ):
        result = asyncio.run(client.ask_async("Question"))

    assert result["answer"] == "async"
