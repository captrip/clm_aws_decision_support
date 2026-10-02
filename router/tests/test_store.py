import threading

import numpy as np
import pytest

from router import config
from router.store import Store

ARM = config.ARM_NAMES[0]


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    yield s
    s.close()


def make_vector(fill=0.5):
    return np.full(config.CONTEXT_DIM, fill)


def log_one(store, **overrides):
    kwargs = dict(
        conversation_id="conv-1",
        turn_index=0,
        arm=ARM,
        context_vector=make_vector(),
        retry_depth=0,
        task_type="prose",
        cost=0.001,
        reward=None,
        needs_judge=True,
        query="hello",
        response_text="hi there",
    )
    kwargs.update(overrides)
    return store.log_decision(**kwargs)


def test_log_decision_returns_increasing_ids(store):
    first = log_one(store)
    second = log_one(store)
    assert isinstance(first, int)
    assert second > first


def test_decision_round_trips_including_the_vector(store):
    vector = np.linspace(0, 1, config.CONTEXT_DIM)
    decision_id = log_one(store, context_vector=vector, cost=0.0042)
    row = store.get_decision(decision_id)
    assert row["arm"] == ARM
    assert row["cost"] == pytest.approx(0.0042)
    assert row["task_type"] == "prose"
    assert row["reward"] is None
    np.testing.assert_allclose(row["context_vector"], vector)
    assert row["context_vector"].shape == (config.CONTEXT_DIM,)


def test_get_decision_returns_none_for_an_unknown_id(store):
    assert store.get_decision(9999) is None


def test_attach_reward_sets_the_reward_and_clears_the_judge_flag(store):
    decision_id = log_one(store, needs_judge=True)
    store.attach_reward(decision_id, 1)
    row = store.get_decision(decision_id)
    assert row["reward"] == 1
    assert row["needs_judge"] == 0


def test_attach_reward_on_an_unknown_id_raises(store):
    with pytest.raises(KeyError):
        store.attach_reward(4242, 1)


def test_pending_judge_returns_only_flagged_unrewarded_rows(store):
    needs = log_one(store, needs_judge=True)
    log_one(store, needs_judge=False)
    already = log_one(store, needs_judge=True, reward=1)
    pending_ids = [row["id"] for row in store.pending_judge(limit=10)]
    assert needs in pending_ids
    assert already not in pending_ids
    assert len(pending_ids) == 1


def test_pending_judge_respects_the_limit(store):
    for _ in range(5):
        log_one(store, needs_judge=True)
    assert len(store.pending_judge(limit=2)) == 2


def test_pending_judge_includes_the_vector_for_a_later_update(store):
    vector = make_vector(0.25)
    log_one(store, needs_judge=True, context_vector=vector)
    row = store.pending_judge(limit=1)[0]
    np.testing.assert_allclose(row["context_vector"], vector)
    assert row["arm"] == ARM
    assert row["query"] == "hello"
    assert row["response_text"] == "hi there"


def test_spend_starts_at_zero_for_an_unknown_conversation(store):
    assert store.get_spent("never-seen") == 0.0


def test_add_spend_accumulates(store):
    store.add_spend("conv-9", 0.001)
    store.add_spend("conv-9", 0.002)
    assert store.get_spent("conv-9") == pytest.approx(0.003)


def test_spend_is_tracked_per_conversation(store):
    store.add_spend("conv-a", 0.005)
    store.add_spend("conv-b", 0.001)
    assert store.get_spent("conv-a") == pytest.approx(0.005)
    assert store.get_spent("conv-b") == pytest.approx(0.001)


def test_state_survives_reopening(tmp_path):
    path = str(tmp_path / "persist.db")
    first = Store(path)
    decision_id = first.log_decision(
        conversation_id="c",
        turn_index=1,
        arm=ARM,
        context_vector=make_vector(0.75),
        retry_depth=0,
        task_type="code",
        cost=0.01,
        reward=1,
        needs_judge=False,
        query="q",
        response_text="r",
    )
    first.add_spend("c", 0.01)
    first.close()

    second = Store(path)
    row = second.get_decision(decision_id)
    assert row["reward"] == 1
    assert row["task_type"] == "code"
    np.testing.assert_allclose(row["context_vector"], make_vector(0.75))
    assert second.get_spent("c") == pytest.approx(0.01)
    second.close()


def test_rejects_a_wrong_length_vector(store):
    with pytest.raises(ValueError, match="context_vector"):
        log_one(store, context_vector=np.zeros(7))


def test_pending_judge_skips_rows_from_another_projection_version(store, monkeypatch):
    """PROJECTION_VERSION guards the snapshot but not the vectors already in
    SQLite. The documented reset -- bump the version, delete the snapshot --
    leaves the DB behind, so the next judge batch drains a queue of pre-bump
    vectors into the fresh bandit and corrupts the reset it was meant to be."""
    stale = log_one(store, needs_judge=True)
    monkeypatch.setattr(config, "PROJECTION_VERSION", "pca-v2")
    fresh = log_one(store, needs_judge=True)

    pending_ids = [row["id"] for row in store.pending_judge(limit=10)]
    assert stale not in pending_ids
    assert pending_ids == [fresh]


def test_a_decision_records_the_projection_version_it_was_built_under(store):
    row = store.get_decision(log_one(store))
    assert row["projection_version"] == config.PROJECTION_VERSION


def test_concurrent_writers_lose_no_rows_and_no_spend(store):
    """All three FastAPI handlers are sync `def`, so Starlette runs them on its
    anyio threadpool and concurrent requests share this one connection. Without
    the mutex sqlite3 drops rows, undercounts spend, and raises from inside
    _row_to_dict -- which would feed a garbage vector to the bandit."""
    threads_count, iterations = 4, 50
    errors = []

    def writer(worker):
        for i in range(iterations):
            try:
                log_one(store, conversation_id=f"conv-{worker}", turn_index=i)
                store.add_spend("shared", 0.001)
            except Exception as exc:  # noqa: BLE001 - collected, not swallowed
                errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(w,)) for w in range(threads_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    total = store.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    assert total == threads_count * iterations
    assert store.get_spent("shared") == pytest.approx(
        threads_count * iterations * 0.001
    )
