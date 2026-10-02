import pytest

from router import arms, config

ARM = config.ARM_NAMES[0]


class FakeBedrock:
    """Records converse() kwargs and replays scripted responses."""

    def __init__(self, responses=None, errors=None):
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        if self.responses:
            return self.responses.pop(0)
        return make_response("ok")


class ThrottleError(Exception):
    """Stands in for botocore ThrottlingException."""

    def __init__(self):
        super().__init__("ThrottlingException: slow down")
        self.response = {"Error": {"Code": "ThrottlingException"}}


def make_response(text, input_tokens=100, output_tokens=200, blocks=None):
    content = blocks if blocks is not None else [{"text": text}]
    return {
        "output": {"message": {"content": content}},
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


# --- token_cost

def test_token_cost_uses_the_bedrock_usage_keys():
    usage = {"inputTokens": 1_000_000, "outputTokens": 0}
    assert arms.token_cost(ARM, usage) == pytest.approx(config.ARMS[ARM]["input_cost"] * 1e6)


def test_token_cost_sums_input_and_output():
    usage = {"inputTokens": 1000, "outputTokens": 2000}
    expected = (
        1000 * config.ARMS[ARM]["input_cost"] + 2000 * config.ARMS[ARM]["output_cost"]
    )
    assert arms.token_cost(ARM, usage) == pytest.approx(expected)


def test_token_cost_of_missing_usage_is_zero():
    assert arms.token_cost(ARM, {}) == 0.0
    assert arms.token_cost(ARM, None) == 0.0


def test_token_cost_of_an_unknown_arm_raises():
    with pytest.raises(KeyError):
        arms.token_cost("not-a-model", {"inputTokens": 1, "outputTokens": 1})


def test_expensive_arm_costs_more_for_identical_usage():
    usage = {"inputTokens": 1000, "outputTokens": 1000}
    cheap = arms.token_cost(config.ARM_NAMES[0], usage)
    dear = arms.token_cost(config.ARM_NAMES[-1], usage)
    assert dear > cheap


# --- bedrock_converse

def test_returns_text_and_usage():
    client = FakeBedrock([make_response("four", 10, 20)])
    text, usage = arms.bedrock_converse(client, ARM, "what is 2+2")
    assert text == "four"
    assert usage["inputTokens"] == 10
    assert usage["outputTokens"] == 20


def test_prompt_is_sent_as_a_user_message():
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hello")
    call = client.calls[0]
    assert call["modelId"] == ARM
    assert call["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]


def test_temperature_is_omitted_unless_requested():
    """Sonnet 5 rejects temperature in inferenceConfig."""
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hi")
    assert "temperature" not in client.calls[0]["inferenceConfig"]


def test_temperature_is_sent_when_requested():
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hi", temperature=0.0)
    assert client.calls[0]["inferenceConfig"]["temperature"] == 0.0


def test_disable_thinking_sets_the_additional_field():
    """Otherwise Sonnet 5 spends the whole budget on reasoningContent."""
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hi", disable_thinking=True)
    extra = client.calls[0]["additionalModelRequestFields"]
    assert extra == {"thinking": {"type": "disabled"}}


def test_thinking_field_absent_by_default():
    client = FakeBedrock()
    arms.bedrock_converse(client, ARM, "hi")
    assert "additionalModelRequestFields" not in client.calls[0]


def test_text_is_joined_across_all_content_blocks():
    """Never assume content[0] holds the text -- reasoning models put a
    reasoningContent block first."""
    blocks = [
        {"reasoningContent": {"reasoningText": {"text": "hmm"}}},
        {"text": "part one "},
        {"text": "part two"},
    ]
    client = FakeBedrock([make_response("", blocks=blocks)])
    text, _ = arms.bedrock_converse(client, ARM, "hi")
    assert text == "part one part two"


def test_empty_content_yields_an_empty_string():
    client = FakeBedrock([make_response("", blocks=[])])
    text, usage = arms.bedrock_converse(client, ARM, "hi")
    assert text == ""
    assert usage["outputTokens"] == 200


def test_throttling_is_retried_with_backoff():
    sleeps = []
    client = FakeBedrock(
        responses=[make_response("eventually")],
        errors=[ThrottleError(), ThrottleError()],
    )
    text, _ = arms.bedrock_converse(
        client, ARM, "hi", sleep_fn=sleeps.append
    )
    assert text == "eventually"
    assert len(client.calls) == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # exponential


def test_throttling_gives_up_after_max_retries():
    client = FakeBedrock(errors=[ThrottleError() for _ in range(10)])
    with pytest.raises(Exception, match="Throttl"):
        arms.bedrock_converse(
            client, ARM, "hi", max_retries=3, sleep_fn=lambda _: None
        )
    assert len(client.calls) == 3


def test_a_non_throttling_error_is_not_retried():
    class Boom(Exception):
        pass

    client = FakeBedrock(errors=[Boom("bad request")])
    with pytest.raises(Boom):
        arms.bedrock_converse(client, ARM, "hi", sleep_fn=lambda _: None)
    assert len(client.calls) == 1


def test_zero_max_retries_raises_a_clear_error():
    """Guards `raise None`: an empty retry loop must not surface a TypeError."""
    client = FakeBedrock()
    with pytest.raises(ValueError, match="max_retries"):
        arms.bedrock_converse(client, ARM, "hi", max_retries=0)
    assert client.calls == []
