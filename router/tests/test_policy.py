import numpy as np
import pytest

from router import arms, config
from router.bandit import LinUCB
from router.context import ContextBuilder
from router.cost import CostEstimator
from router.policy import Router
from router.store import Store


class FakeEmbedder:
    def encode(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        return rng.normal(size=config.EMBED_DIM)


class RecordingBandit(LinUCB):
    """LinUCB that logs every (arm, vector, reward) update."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.updates = []

    def update(self, arm, x, reward):
        self.updates.append((arm, np.array(x, copy=True), reward))
        super().update(arm, x, reward)


def make_router(
    converse_fn, verify_fn=None, tmp_path=None, forced_arms=None, observer=None
):
    bandit = RecordingBandit(
        arm_names=config.ARM_NAMES,
        dim=config.CONTEXT_DIM,
        alpha=config.ALPHA,
        lambda_reg=config.LAMBDA_REG,
        projection_version=config.PROJECTION_VERSION,
    )
    if forced_arms is not None:
        sequence = list(forced_arms)

        def fake_select(x, candidates, costs, remaining_budget):
            while sequence:
                arm = sequence.pop(0)
                if arm in candidates:
                    return arm, bandit.score(x)
            return None, bandit.score(x)

        bandit.select = fake_select

    store = Store(str(tmp_path / "policy.db"))
    router = Router(
        bandit=bandit,
        context_builder=ContextBuilder(embedder=FakeEmbedder()),
        cost_estimator=CostEstimator(),
        store=store,
        client=object(),
        converse_fn=converse_fn,
        verify_fn=verify_fn if verify_fn is not None else (lambda *a, **k: None),
        observer=observer,
    )
    return router, bandit, store


def usage(input_tokens=100, output_tokens=100):
    return {"inputTokens": input_tokens, "outputTokens": output_tokens}


def test_single_call_when_the_verifier_passes(tmp_path):
    calls = []

    def converse(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        return "answer", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(calls) == 1
    assert result["cascade_depth"] == 1
    assert result["status"] == "success"
    assert result["verdict"] == 1
    assert bandit.updates[0][2] == 1
    store.close()


def test_escalates_when_the_verifier_fails(tmp_path):
    calls = []
    verdicts = [0, 0, 1]

    def converse(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        return f"answer from {model_id}", usage()

    def verifier(*args, **kwargs):
        return verdicts.pop(0)

    router, bandit, store = make_router(converse, verifier, tmp_path)
    result = router.complete("conv", "q", task_type="code")
    assert len(calls) == 3
    assert len(set(calls)) == 3, "must not retry the same arm"
    assert result["cascade_depth"] == 3
    assert result["status"] == "success"
    assert [u[2] for u in bandit.updates] == [0, 0, 1]
    store.close()


def test_does_not_escalate_on_an_unverifiable_verdict(tmp_path):
    calls = []

    def converse(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        return "prose answer", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: None, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="prose")
    assert len(calls) == 1
    assert result["status"] == "unverified"
    assert result["verdict"] is None
    assert bandit.updates == [], "no reward yet, so no update"
    pending = store.pending_judge(limit=10)
    assert len(pending) == 1
    store.close()


def test_stops_at_max_cascade_depth(tmp_path):
    calls = []

    def converse(client, model_id, prompt, **kwargs):
        calls.append(model_id)
        return "wrong", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 0, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(calls) == config.MAX_CASCADE
    assert result["status"] == "exhausted"
    assert result["cascade_depth"] == config.MAX_CASCADE
    store.close()


def test_each_arm_is_updated_with_the_vector_it_saw(tmp_path):
    """The DESIGN.md section 6 rule. Arm 1 saw retry_depth 0 and no failure;
    arm 2 saw retry_depth 1 and arm 1's one-hot. Updating arm 1 with arm 2's
    vector corrupts learning invisibly."""
    verdicts = [0, 1]

    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, lambda *a, **k: verdicts.pop(0), tmp_path
    )
    router.complete("conv", "q", task_type="code")

    assert len(bandit.updates) == 2
    (first_arm, first_vec, _), (second_arm, second_vec, _) = bandit.updates

    # First arm's vector: no prior failure, retry depth 0
    assert first_vec[35] == 0.0
    assert first_vec[36:43].sum() == 0.0
    assert first_vec[47] == 0.0

    # Second arm's vector: failure flag on, first arm's one-hot set, depth 1
    assert second_vec[35] == 1.0
    first_index = config.ARM_NAMES.index(first_arm)
    assert second_vec[36 + first_index] == 1.0
    assert second_vec[47] == pytest.approx(1.0 / config.MAX_CASCADE)
    assert not np.allclose(first_vec, second_vec)
    store.close()


def test_stops_when_the_budget_is_exhausted(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        return "wrong", usage(input_tokens=10 ** 6, output_tokens=10 ** 6)

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 0, tmp_path=tmp_path
    )
    # Every arm's fallback estimate is affordable at 0.05, so one call happens;
    # every arm's actual cost at 1e6/1e6 tokens exceeds 0.05, so the cascade
    # stops after it rather than escalating.
    result = router.complete("conv", "q", task_type="code", budget=0.05)
    assert result["status"] == "budget_exhausted"
    assert result["cascade_depth"] == 1
    assert result["cost"] > 0.05
    store.close()


def test_returns_budget_exhausted_when_nothing_is_affordable(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        raise AssertionError("must not call any model")

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 0, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code", budget=1e-12)
    assert result["status"] == "budget_exhausted"
    assert result["cascade_depth"] == 0
    assert result["text"] == ""
    store.close()


def test_cost_is_accumulated_per_conversation(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    router.complete("conv-x", "q1", task_type="code")
    first = store.get_spent("conv-x")
    router.complete("conv-x", "q2", task_type="code", turn_index=1)
    assert store.get_spent("conv-x") > first
    assert store.get_spent("conv-other") == 0.0
    store.close()


def test_cost_estimator_learns_from_actual_spend(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path,
        forced_arms=[config.ARM_NAMES[0]] * 4,
    )
    arm = config.ARM_NAMES[0]
    before = router.cost_estimator.estimate(arm, 0)
    for _ in range(3):
        router.complete("conv", "q", task_type="code")
    after = router.cost_estimator.estimate(arm, 0)
    assert after != before
    store.close()


def test_every_attempt_is_logged_with_a_decision_id(tmp_path):
    verdicts = [0, 1]

    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, lambda *a, **k: verdicts.pop(0), tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(result["decision_ids"]) == 2
    for decision_id in result["decision_ids"]:
        assert store.get_decision(decision_id) is not None
    store.close()


def test_a_model_error_is_recorded_and_escalates(tmp_path):
    attempts = []

    def converse(client, model_id, prompt, **kwargs):
        attempts.append(model_id)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return "recovered", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(attempts) == 2
    assert result["status"] == "success"
    assert result["text"] == "recovered"
    # The erroring arm is NOT penalised: an infrastructure failure must not
    # teach the bandit that a good arm is weak. Only the arm that actually
    # answered gets an update.
    assert len(bandit.updates) == 1
    assert bandit.updates[0][0] == attempts[1]
    assert bandit.updates[0][2] == 1
    store.close()


def test_an_error_row_stores_no_reward_and_is_not_for_the_judge(tmp_path):
    """A/b are exactly reconstructible from the rows with a non-null reward, so
    persisting reward=0 for an arm the bandit deliberately did not learn from
    means any rebuild-from-DB reapplies precisely the penalty that was
    suppressed. record_reward cannot correct it either -- reward is not null, so
    it returns False. The escalation itself must not change."""
    attempts = []

    def converse(client, model_id, prompt, **kwargs):
        attempts.append(model_id)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return "recovered", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")

    error_row = store.get_decision(result["decision_ids"][0])
    assert error_row["reward"] is None
    assert error_row["needs_judge"] == 0
    assert [row["id"] for row in store.pending_judge(limit=10)] == []

    # Control flow untouched: it still escalated and the answering arm learned.
    assert len(attempts) == 2
    assert result["status"] == "success"
    assert store.get_decision(result["decision_ids"][1])["reward"] == 1
    store.close()


def test_record_reward_refuses_an_infrastructure_error_row(tmp_path):
    """Storing reward=NULL for an error row means NULL alone no longer means
    "awaiting feedback". /v1/complete returns every decision_id, so a client
    rating all of them would otherwise apply exactly the penalty complete()
    suppressed -- to an arm whose response_text is empty."""
    attempts = []

    def converse(client, model_id, prompt, **kwargs):
        attempts.append(model_id)
        if len(attempts) == 1:
            raise RuntimeError("model unavailable")
        return "recovered", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    error_id = result["decision_ids"][0]
    before = list(bandit.updates)

    assert router.record_reward(error_id, 1) is False
    assert bandit.updates == before
    # And the row still carries no reward, so a rebuild-from-DB stays clean.
    assert store.get_decision(error_id)["reward"] is None
    store.close()


def test_an_erroring_arm_records_no_cost_sample(tmp_path):
    """An error observed no cost, so 0.0 is not a cost sample. Recording it
    drives the LCB to EPSILON_COST -- six orders of magnitude under the arm's
    real price -- while learn=False keeps its full initial optimism, so the
    broken arm becomes a permanent first pick that cannot be escaped."""
    cheap = config.ARM_NAMES[0]

    def converse(client, model_id, prompt, **kwargs):
        if model_id == cheap:
            raise RuntimeError("AccessDeniedException")
        return "answer", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    before = router.cost_estimator.estimate(cheap, 0)
    for _ in range(5):
        router.complete("conv", "q", turn_index=0, task_type="code")
    assert router.cost_estimator.estimate(cheap, 0) == pytest.approx(before)

    # And with no bogus sample, the broken arm stops being the automatic pick.
    served = [
        router.complete("conv2", "q", task_type="code")["arm"] for _ in range(5)
    ]
    assert any(arm != cheap for arm in served)
    store.close()


def test_record_reward_updates_the_bandit_with_the_stored_vector(tmp_path):
    def converse(client, model_id, prompt, **kwargs):
        return "prose", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: None, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="prose")
    decision_id = result["decision_ids"][0]
    assert bandit.updates == []

    router.record_reward(decision_id, 1)
    assert len(bandit.updates) == 1
    arm, vector, reward = bandit.updates[0]
    stored = store.get_decision(decision_id)
    assert arm == stored["arm"]
    assert reward == 1
    np.testing.assert_allclose(vector, stored["context_vector"])
    assert stored["reward"] == 1
    store.close()


def test_record_reward_refuses_a_vector_from_another_projection_version(
    tmp_path, monkeypatch
):
    """After the documented reset the DB still holds pre-bump vectors. Applying
    one writes old-feature-space values into the new bandit's b, which is worse
    than dropping the reward: it silently undoes the reset."""

    def converse(client, model_id, prompt, **kwargs):
        return "prose", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: None, tmp_path=tmp_path
    )
    decision_id = router.complete("conv", "q", task_type="prose")["decision_ids"][0]

    monkeypatch.setattr(config, "PROJECTION_VERSION", "pca-v2")
    assert router.record_reward(decision_id, 1) is False
    assert bandit.updates == []
    assert store.get_decision(decision_id)["reward"] is None
    store.close()


def test_record_reward_is_idempotent(tmp_path):
    """A retried webhook or a double judge pass must not update twice: two
    updates from one observation double-count it and skew that arm."""

    def converse(client, model_id, prompt, **kwargs):
        return "prose", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: None, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="prose")
    decision_id = result["decision_ids"][0]
    assert router.record_reward(decision_id, 1) is True
    assert router.record_reward(decision_id, 1) is False
    assert len(bandit.updates) == 1
    store.close()


def test_record_reward_ignores_an_already_verified_decision(tmp_path):
    """complete() already updated the bandit for a verified turn, so a later
    /v1/reward for it must be a no-op rather than a second update."""

    def converse(client, model_id, prompt, **kwargs):
        return "a", usage()

    router, bandit, store = make_router(
        converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    result = router.complete("conv", "q", task_type="code")
    assert len(bandit.updates) == 1
    assert router.record_reward(result["decision_ids"][0], 0) is False
    assert len(bandit.updates) == 1
    store.close()


class _RecordingObserver:
    """Captures events. next_turn_id mirrors viz.VizObserver's contract."""

    def __init__(self):
        self.events = []
        self._turn = 0

    def next_turn_id(self):
        self._turn += 1
        return self._turn

    def emit(self, event, payload):
        self.events.append((event, payload))

    def names(self):
        return [name for name, _ in self.events]

    def payload(self, name):
        for event_name, payload in self.events:
            if event_name == name:
                return payload
        raise AssertionError(f"no {name} event in {self.names()}")


class _ExplodingObserver:
    def next_turn_id(self):
        raise RuntimeError("boom in next_turn_id")

    def emit(self, event, payload):
        raise RuntimeError("boom in emit")


def _ok_converse(client, model_id, prompt, **kwargs):
    return "an answer", usage()


def test_emit_is_a_noop_without_an_observer(tmp_path):
    router, _bandit, store = make_router(_ok_converse, tmp_path=tmp_path)
    assert router.observer is None
    router._emit("anything", a=1)  # must not raise
    store.close()


def test_an_observer_receives_events(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    router._emit("ping", a=1)
    assert observer.events == [("ping", {"a": 1})]
    store.close()


def test_an_exploding_observer_cannot_break_emit(tmp_path):
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=_ExplodingObserver()
    )
    router._emit("ping", a=1)  # must not raise
    store.close()


def test_an_exploding_observer_cannot_break_a_turn(tmp_path):
    # The load-bearing test of this whole feature. A visualization that can take
    # down routing is worse than no visualization. Note the observer raises from
    # next_turn_id too, so the turn_id lookup must be guarded as well as _emit.
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path,
        observer=_ExplodingObserver(),
    )
    result = router.complete("conv", "q", task_type="code")
    assert result["status"] == "success"
    assert result["arm"] is not None
    store.close()


def test_observer_none_leaves_the_result_unchanged(tmp_path):
    # Same stubs, fresh identical bandits, with and without an observer:
    # identical results.
    plain, _b1, store1 = make_router(
        _ok_converse, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path
    )
    plain_result = plain.complete("conv", "q", task_type="code")
    store1.close()

    (tmp_path / "watched").mkdir()
    watched, _b2, store2 = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path / "watched",
        observer=_RecordingObserver(),
    )
    watched_result = watched.complete("conv", "q", task_type="code")
    store2.close()

    for key in ("arm", "status", "cascade_depth", "verdict", "text"):
        assert plain_result[key] == watched_result[key], key


def test_a_verified_turn_emits_the_full_event_sequence(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path,
        observer=observer,
    )
    router.complete("conv", "q", task_type="code")
    assert observer.names() == [
        "turn_start",
        "scored",
        "called",
        "verified",
        "updated",
        "turn_end",
    ]
    store.close()


def test_called_carries_this_attempts_real_cost(tmp_path):
    """The per-attempt cost must be in the event, not just turn_end's total.

    It used to emit cost=None because token_cost was computed after the emit,
    so the UI could only ever show the turn aggregate -- a 3-attempt cascade
    gave no way to see which attempt spent what. token_cost is a pure function
    of (arm, usage), so computing it before the emit changes no accounting.
    """
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path,
        observer=observer,
    )
    result = router.complete("conv", "q", task_type="code")
    called = observer.payload("called")
    expected = arms.token_cost(called["arm"], called["usage"])
    assert called["cost"] == expected
    assert called["cost"] > 0.0
    # Single-attempt turn, so the attempt cost IS the turn cost.
    assert called["cost"] == pytest.approx(result["cost"])
    store.close()


def test_called_reports_zero_cost_on_an_infrastructure_error(tmp_path):
    # An error observed no usage, so it truly cost nothing. The event must say
    # 0.0 rather than inventing a charge -- and must not be None either, or the
    # UI cannot distinguish "free" from "unknown".
    observer = _RecordingObserver()

    def boom(client, model_id, prompt, **kwargs):
        raise RuntimeError("ThrottlingException: rate exceeded")

    router, _bandit, store = make_router(
        boom, verify_fn=lambda *a, **k: 1, tmp_path=tmp_path, observer=observer
    )
    router.complete("conv", "q", task_type="code")
    called = observer.payload("called")
    assert called["cost"] == 0.0
    assert called["error"] is not None
    store.close()


def test_scored_carries_every_arm_including_ineligible_ones(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 1,
        tmp_path=tmp_path,
        observer=observer,
    )
    router.complete("conv", "q", task_type="code")
    scores = observer.payload("scored")["scores"]
    assert set(scores) == set(config.ARM_NAMES)
    for info in scores.values():
        assert {
            "p",
            "e",
            "ucb",
            "cost_estimate",
            "eligible",
            "reason",
            "selection_score",
        } == set(info)
    assert observer.payload("scored")["chosen"] in config.ARM_NAMES
    store.close()


def test_an_unverifiable_turn_emits_no_update_and_ends_unverified(tmp_path):
    observer = _RecordingObserver()
    # make_router's default verify_fn returns None -- the "cannot tell" path.
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    router.complete("conv", "q", task_type="prose")
    assert "updated" not in observer.names()
    assert observer.payload("verified")["verdict"] is None
    assert observer.payload("verified")["learn"] is False
    assert observer.payload("turn_end")["status"] == "unverified"
    store.close()


def test_a_cascade_emits_one_scored_per_attempt(tmp_path):
    observer = _RecordingObserver()
    # A verifier that always returns 0 escalates every attempt, so the cascade
    # runs to MAX_CASCADE (3) and exhausts.
    router, _bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 0,
        tmp_path=tmp_path,
        observer=observer,
    )
    router.complete("conv", "q", task_type="code")
    depths = [p["depth"] for name, p in observer.events if name == "scored"]
    assert depths == [0, 1, 2]
    assert observer.payload("turn_end")["status"] == "exhausted"
    assert observer.payload("turn_end")["cascade_depth"] == 3
    store.close()


def test_updated_uses_the_vector_the_arm_was_selected_with(tmp_path):
    # The router's core attribution rule (DESIGN.md section 6): complete()
    # rebuilds x per attempt, so attempt 2's vector has the failure flag set and
    # retry_depth incremented. Reporting attempt 2's vector for attempt 1's arm
    # would misstate the update. Each update's before-values must equal what
    # that same arm scored at its own selection moment.
    observer = _RecordingObserver()
    router, bandit, store = make_router(
        _ok_converse,
        verify_fn=lambda *a, **k: 0,
        tmp_path=tmp_path,
        observer=observer,
    )
    router.complete("conv", "q", task_type="code")
    scored = [p for name, p in observer.events if name == "scored"]
    updated = [p for name, p in observer.events if name == "updated"]
    assert len(updated) >= 2, "need at least two cascade steps to distinguish"
    for step in updated:
        matching = [s for s in scored if s["depth"] == step["depth"]]
        assert len(matching) == 1
        chosen = matching[0]["chosen"]
        assert step["arm"] == chosen
        assert step["before"]["ucb"] == pytest.approx(
            matching[0]["scores"][chosen]["ucb"]
        )
        assert step["before"]["p"] == pytest.approx(
            matching[0]["scores"][chosen]["p"]
        )
    # And cross-check against RecordingBandit, which logs the real vectors:
    # the emitted arms must match the arms actually updated, in order.
    assert [s["arm"] for s in updated] == [u[0] for u in bandit.updates]
    store.close()


def test_an_infrastructure_error_emits_called_with_the_error_and_does_not_update(
    tmp_path,
):
    observer = _RecordingObserver()

    def exploding_converse(client, model_id, prompt, **kwargs):
        raise RuntimeError("bedrock is down")

    router, _bandit, store = make_router(
        exploding_converse, tmp_path=tmp_path, observer=observer
    )
    router.complete("conv", "q", task_type="code")
    assert observer.payload("called")["error"] is not None
    assert "updated" not in observer.names()
    # verdict is 0 (escalate) but learn is False (environmental fault).
    assert observer.payload("verified")["verdict"] == 0
    assert observer.payload("verified")["learn"] is False
    store.close()


def test_turn_end_fires_when_no_arm_is_affordable(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    # A budget below every arm's estimate prunes all of them, so select()
    # returns None on the first attempt -- verified: every arm's reason is
    # "over budget" at 1e-9.
    router.complete("conv", "q", task_type="code", budget=1e-9)
    assert observer.payload("turn_end")["status"] == "budget_exhausted"
    assert observer.payload("turn_end")["cascade_depth"] == 0
    assert "called" not in observer.names()
    scores = observer.payload("scored")["scores"]
    assert all(not s["eligible"] for s in scores.values())
    store.close()


def test_record_reward_emits_delayed_update_when_applied(tmp_path):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    result = router.complete("conv", "q", task_type="prose")
    decision_id = result["decision_ids"][0]
    assert router.record_reward(decision_id, 1) is True
    payload = observer.payload("delayed_update")
    assert payload["applied"] is True
    assert payload["decision_id"] == decision_id
    assert payload["pulls_after"] == payload["pulls_before"] + 1
    assert payload["reward"] == 1
    store.close()


def test_record_reward_emits_delayed_update_with_applied_false_on_a_repeat(
    tmp_path,
):
    observer = _RecordingObserver()
    router, _bandit, store = make_router(
        _ok_converse, tmp_path=tmp_path, observer=observer
    )
    result = router.complete("conv", "q", task_type="prose")
    decision_id = result["decision_ids"][0]
    router.record_reward(decision_id, 1)
    assert router.record_reward(decision_id, 1) is False
    delayed = [p for name, p in observer.events if name == "delayed_update"]
    assert [p["applied"] for p in delayed] == [True, False]
    assert delayed[1]["pulls_before"] == delayed[1]["pulls_after"]
    assert delayed[1]["reason"] == "already_rewarded"
    store.close()
