"""Bedrock converse wrapper and cost accounting.

Adapted from bedrock_converse in Algorithm/Algo1_AWS.ipynb, including the two
Sonnet 5 quirks documented in README_AWS.md:
  - temperature is only sent when a caller asks for one (Sonnet 5 rejects it)
  - text is joined across all content blocks, never content[0], because a
    reasoning model puts a reasoningContent block first
"""

import time

from router import config


def make_client(profile=config.AWS_PROFILE, region=config.AWS_REGION):
    import boto3

    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("bedrock-runtime", region_name=region)


def _is_throttling(error):
    code = ""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
    return "Throttl" in code or "Throttl" in str(error)


def bedrock_converse(
    client,
    model_id,
    prompt,
    temperature=None,
    max_tokens=1024,
    disable_thinking=False,
    max_retries=5,
    sleep_fn=time.sleep,
):
    """Call converse and return (text, usage). Retries throttling only."""
    request = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    if temperature is not None:
        request["inferenceConfig"]["temperature"] = temperature
    if disable_thinking:
        request["additionalModelRequestFields"] = {"thinking": {"type": "disabled"}}

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.converse(**request)
            blocks = response.get("output", {}).get("message", {}).get("content", [])
            text = "".join(
                block["text"] for block in blocks if isinstance(block.get("text"), str)
            )
            usage = response.get("usage", {}) or {}
            return text, usage
        except Exception as error:  # noqa: BLE001 - re-raised below
            if not _is_throttling(error):
                raise
            last_error = error
            if attempt < max_retries - 1:
                sleep_fn(2.0 ** attempt)
    if last_error is None:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    raise last_error


def token_cost(model_id, usage):
    """Cost in USD from Bedrock's inputTokens / outputTokens."""
    cfg = config.ARMS[model_id]
    if not usage:
        return 0.0
    return (
        usage.get("inputTokens", 0) * cfg["input_cost"]
        + usage.get("outputTokens", 0) * cfg["output_cost"]
    )
