"""AWS Bedrock Converse adapter."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from .models import InvocationResult, ModelSpec, ProviderKind, Usage


_TRANSIENT_ERROR_CODES = frozenset(
    {
        "InternalServerException",
        "ModelNotReadyException",
        "ServiceUnavailableException",
        "ThrottlingException",
    }
)


class BedrockProvider:
    """Invoke Bedrock models through boto3's Converse API."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_attempts: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self._client = client
        self._max_attempts = max_attempts
        self._sleeper = sleeper

    def invoke(
        self,
        model: ModelSpec,
        prompt: str,
        max_tokens: int,
        temperature: float | None,
        additional_fields: Mapping[str, Any] | None = None,
    ) -> InvocationResult:
        if model.provider is not ProviderKind.BEDROCK:
            raise ValueError("BedrockProvider requires a bedrock model")

        request: dict[str, Any] = {
            "modelId": model.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if temperature is not None:
            request["inferenceConfig"]["temperature"] = temperature
        if additional_fields is not None:
            request["additionalModelRequestFields"] = dict(additional_fields)

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._get_client().converse(**request)
                return InvocationResult(
                    text=_response_text(response),
                    provider=ProviderKind.BEDROCK,
                    usage=Usage(
                        input_tokens=int(response.get("usage", {}).get("inputTokens", 0)),
                        output_tokens=int(response.get("usage", {}).get("outputTokens", 0)),
                    ),
                )
            except Exception as error:
                if attempt == self._max_attempts or not _is_transient(error):
                    raise
                self._sleeper(0.25 * attempt)

        raise AssertionError("unreachable")

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime")
        return self._client


def _response_text(response: Mapping[str, Any]) -> str:
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    return "".join(block["text"] for block in blocks if "text" in block)


def _is_transient(error: Exception) -> bool:
    response = getattr(error, "response", {})
    code = response.get("Error", {}).get("Code") if isinstance(response, Mapping) else None
    return code in _TRANSIENT_ERROR_CODES
