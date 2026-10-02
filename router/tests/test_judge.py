import numpy as np
import pytest

from router import config, judge


class StubStore:
    def __init__(self, rows):
        self._rows = rows

    def pending_judge(self, limit):
        return self._rows[:limit]


class StubRouter:
    def __init__(self, rows):
        self.store = StubStore(rows)
        self.rewards = []

    def record_reward(self, decision_id, reward):
        self.rewards.append((decision_id, reward))
        return True


def make_row(decision_id=1, query="q", response_text="r"):
    return {
        "id": decision_id,
        "arm": config.ARM_NAMES[0],
        "query": query,
        "response_text": response_text,
        "context_vector": np.zeros(config.CONTEXT_DIM),
    }


class AlwaysTrue:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def __call__(self, client, model_id, prompt, **kwargs):
        self.calls.append({"model_id": model_id, "prompt": prompt, **kwargs})
        reply = self.replies.pop(0) if self.replies else "True"
        return reply, {"inputTokens": 10, "outputTokens": 1}


# --- prompt and parsing

def test_prompt_contains_the_query_and_the_response():
    prompt = judge.build_judge_prompt("what is 2+2", "four")
    assert "what is 2+2" in prompt
    assert "four" in prompt
    assert "True" in prompt and "False" in prompt


def test_parse_verdict_reads_true_and_false_case_insensitively():
    assert judge.parse_verdict("True") == 1
    assert judge.parse_verdict("  true\n") == 1
    assert judge.parse_verdict("FALSE") == 0
    assert judge.parse_verdict("false.") == 0


def test_parse_verdict_returns_none_when_ambiguous():
    assert judge.parse_verdict("maybe") is None
    assert judge.parse_verdict("") is None
    assert judge.parse_verdict(None) is None


def test_parse_verdict_requires_a_whole_word():
    """"untrue" read as a substring would score a bad answer as good."""
    assert judge.parse_verdict("untrue") is None
    assert judge.parse_verdict("construe") is None
    assert judge.parse_verdict("falsehood") is None


def test_parse_verdict_treats_both_words_as_ambiguous():
    """A hedging reply is "cannot tell", not a vote for whichever came first."""
    assert judge.parse_verdict("False, that is not true") is None
    assert judge.parse_verdict("True or false") is None


def test_parse_verdict_requires_the_verdict_to_lead_the_reply():
    """The prompt asks for only the word. A negation must not read as a vote:
    "not true" contains the whole word "true", so word boundaries alone are
    not enough."""
    assert judge.parse_verdict("not true") is None
    assert judge.parse_verdict("that is false") is None
    assert judge.parse_verdict("'True'") == 1


# --- batch

def test_judged_rows_get_a_reward():
    router = StubRouter([make_row(5)])
    converse = AlwaysTrue(["True"])
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    assert router.rewards == [(5, 1)]
    assert stats["judged"] == 1
    assert stats["considered"] == 1


def test_a_false_verdict_records_reward_zero():
    router = StubRouter([make_row(6)])
    converse = AlwaysTrue(["False"])
    judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    assert router.rewards == [(6, 0)]


def test_an_ambiguous_verdict_records_nothing():
    router = StubRouter([make_row(7)])
    converse = AlwaysTrue(["I am not sure"])
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    assert router.rewards == []
    assert stats["ambiguous"] == 1
    assert stats["judged"] == 0


def test_sample_rate_zero_judges_nothing():
    router = StubRouter([make_row(i) for i in range(10)])
    converse = AlwaysTrue()
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=0.0, converse_fn=converse
    )
    assert router.rewards == []
    assert converse.calls == []
    assert stats["skipped"] == 10


def test_sample_rate_is_applied_with_the_supplied_rng():
    router = StubRouter([make_row(i) for i in range(100)])
    converse = AlwaysTrue()
    stats = judge.run_judge_batch(
        router,
        client=object(),
        limit=100,
        sample_rate=0.5,
        converse_fn=converse,
        rng=np.random.default_rng(0),
    )
    assert 20 < stats["judged"] < 80
    assert stats["judged"] + stats["skipped"] == 100


def test_the_judge_uses_sonnet_with_thinking_disabled():
    """Sonnet 5 otherwise spends the whole token budget on reasoningContent
    and returns no text at all (README_AWS.md)."""
    router = StubRouter([make_row(1)])
    converse = AlwaysTrue(["True"])
    judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    call = converse.calls[0]
    assert call["model_id"] == config.JUDGE_MODEL
    assert call["disable_thinking"] is True
    assert "temperature" not in call or call["temperature"] is None


def test_a_judge_error_does_not_abort_the_batch():
    calls = []

    def flaky(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        if len(calls) == 1:
            raise RuntimeError("judge unavailable")
        return "True", {"inputTokens": 1, "outputTokens": 1}

    router = StubRouter([make_row(1), make_row(2)])
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=flaky
    )
    assert router.rewards == [(2, 1)]
    assert stats["errors"] == 1


def test_an_empty_queue_is_a_no_op():
    router = StubRouter([])
    converse = AlwaysTrue()
    stats = judge.run_judge_batch(
        router, client=object(), sample_rate=1.0, converse_fn=converse
    )
    assert stats == {
        "considered": 0,
        "judged": 0,
        "skipped": 0,
        "ambiguous": 0,
        "errors": 0,
    }
    assert converse.calls == []


def test_the_default_sample_rate_comes_from_config():
    assert config.JUDGE_SAMPLE_RATE == 0.10
