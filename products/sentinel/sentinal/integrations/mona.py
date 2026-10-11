"""Client for routing prompts through the Whitebat MONA API."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MonaPolicy = Literal["linucb", "clm", "hybrid"]


class MonaAPIError(RuntimeError):
    """A MONA API request failed."""


class MonaClient:
    """Call MONA's API-key-authenticated model-selection endpoint.

    Configure ``MONA_API_URL`` and ``MONA_API_KEY`` in the process environment,
    or pass them explicitly. The API key is sent only in the ``X-API-Key``
    header and must be kept secret.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("MONA_API_KEY")
        configured_url = base_url or os.environ.get("MONA_API_URL")
        if not self._api_key:
            raise ValueError("Set MONA_API_KEY or pass api_key explicitly.")
        if not configured_url:
            raise ValueError("Set MONA_API_URL or pass base_url explicitly.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")
        self._endpoint = f"{configured_url.rstrip('/')}/api/ask"
        self._timeout = timeout

    def ask(
        self,
        question: str,
        *,
        policy: MonaPolicy = "linucb",
        budget: float | None = 0.01,
    ) -> dict[str, Any]:
        """Route a prompt through MONA and return its decision and answer."""
        if not question.strip():
            raise ValueError("question must not be empty.")
        if policy not in {"linucb", "clm", "hybrid"}:
            raise ValueError(f"Unsupported MONA policy: {policy!r}.")
        if budget is not None and budget < 0:
            raise ValueError("budget must be zero or greater, or None.")

        body = json.dumps(
            {"question": question, "policy": policy, "budget": budget}
        ).encode("utf-8")
        request = Request(
            self._endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self._api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise MonaAPIError(
                f"MONA request failed with HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise MonaAPIError(f"Could not reach the MONA API: {error.reason}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MonaAPIError("MONA returned an invalid JSON response.") from error

        if not isinstance(result, dict):
            raise MonaAPIError("MONA returned an unexpected response shape.")
        return result

    async def ask_async(
        self,
        question: str,
        *,
        policy: MonaPolicy = "linucb",
        budget: float | None = 0.01,
    ) -> dict[str, Any]:
        """Run :meth:`ask` off the event loop for async agent applications."""
        return await asyncio.to_thread(
            self.ask, question, policy=policy, budget=budget
        )
