"""The only test that touches AWS or downloads a model. Opt in with:
    ROUTER_SMOKE=1 router/.venv/bin/pytest router/tests/test_smoke.py -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ROUTER_SMOKE") != "1",
    reason="set ROUTER_SMOKE=1 to run tests that call AWS and load the embedder",
)


def test_the_real_embedder_produces_a_49_dim_vector():
    from router import config
    from router.context import ContextBuilder

    builder = ContextBuilder()
    x = builder.build(
        query="what is the capital of France",
        turn_index=0,
        prompt_tokens=8,
        conversation_tokens=8,
        task_type="prose",
        failed_arms=[],
        retry_depth=0,
    )
    assert x.shape == (config.CONTEXT_DIM,)
    assert x[48] == 1.0


def test_the_cheapest_arm_answers_over_bedrock():
    from router import arms, config

    client = arms.make_client()
    text, usage = arms.bedrock_converse(
        client,
        config.ARM_NAMES[0],
        "Reply with exactly the word: pong",
        temperature=0.0,
        max_tokens=10,
    )
    assert isinstance(text, str) and text.strip()
    assert usage["inputTokens"] > 0
    assert arms.token_cost(config.ARM_NAMES[0], usage) > 0


def test_the_judge_returns_a_parseable_verdict():
    from router import arms, judge

    client = arms.make_client()
    prompt = judge.build_judge_prompt("what is 2+2", "The answer is 4.")
    text, _ = arms.bedrock_converse(
        client, judge.config.JUDGE_MODEL, prompt, max_tokens=10, disable_thinking=True
    )
    # `is not None`, not `== 1`. This test proves the AWS wiring and the Sonnet
    # quirks: that a real call comes back with something the parser accepts. A
    # live model's exact phrasing is not ours to assert -- parse_verdict now
    # requires the verdict word to lead the reply, so "The answer is true" is a
    # legitimate None and would make this flaky rather than informative.
    assert judge.parse_verdict(text) is not None
