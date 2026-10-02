import json
import pathlib

import pytest

from router import config, replay

# Anchored to this file, not the cwd. A relative "Result/algo1res.json" resolves
# only when pytest runs from the repo root, so it would silently skip elsewhere
# -- and one of the tests below is the sole regression guard for the
# disjoint-arm-name defect, which must never skip quietly.
RESULT_FILE = pathlib.Path(__file__).resolve().parents[2] / "Result" / "algo1res.json"


def make_record(question, truth, attempts):
    return {
        "question": question,
        "ground_truth_answer": truth,
        "final_status": "Success",
        "total_cost": 0.001,
        "attempts": [
            {
                "step": i + 1,
                "chosen_model": arm,
                "is_correct": bool(correct),
                "chosen_model_cost": {"total_cost": 0.0005},
            }
            for i, (arm, correct) in enumerate(attempts)
        ],
    }


def _write(tmp_path, name, records):
    path = tmp_path / name
    path.write_text(json.dumps(records))
    return str(path)


@pytest.fixture
def replay_file(tmp_path):
    path = tmp_path / "records.json"
    records = [
        make_record("q1", "42", [(config.ARM_NAMES[0], 0), (config.ARM_NAMES[1], 1)]),
        make_record("q2", "7", [(config.ARM_NAMES[2], 1)]),
    ]
    path.write_text(json.dumps(records))
    return str(path)


def test_loads_queries_and_per_arm_outcomes(replay_file):
    records = replay.load_replay_records(replay_file)
    assert len(records) == 2
    first = records[0]
    assert first["query"] == "q1"
    assert first["expected"] == "42"
    assert first["arm_outcomes"][config.ARM_NAMES[0]] == 0
    assert first["arm_outcomes"][config.ARM_NAMES[1]] == 1


def test_records_only_contain_arms_that_were_actually_tried(replay_file):
    records = replay.load_replay_records(replay_file)
    assert set(records[1]["arm_outcomes"]) == {config.ARM_NAMES[2]}


def test_the_real_result_file_loads_if_present():
    if not RESULT_FILE.exists():
        pytest.skip(f"{RESULT_FILE} not present")
    records = replay.load_replay_records(str(RESULT_FILE))
    assert len(records) > 100
    assert all("query" in r and "arm_outcomes" in r for r in records)


def test_replay_reports_summary_metrics(replay_file):
    records = replay.load_replay_records(replay_file)
    summary = replay.replay(records, replay.make_offline_router)
    assert summary["n"] == 2
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert summary["total_cost"] >= 0.0
    assert summary["avg_cascade_depth"] >= 1.0
    assert set(summary["arm_counts"]) <= set(config.ARM_NAMES)


def test_replay_honours_max_records(replay_file):
    records = replay.load_replay_records(replay_file)
    summary = replay.replay(records, replay.make_offline_router, max_records=1)
    assert summary["n"] == 1


def test_max_records_zero_replays_nothing(tmp_path):
    """0 is falsy: a truthiness guard would silently replay the whole set."""
    records = replay.load_replay_records(
        _write(tmp_path, "z.json", [make_record("q", "1", [(config.ARM_NAMES[0], 1)])])
    )
    summary = replay.replay(records, replay.make_offline_router, max_records=0)
    assert summary["n"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["avg_cascade_depth"] == 0.0
    assert summary["arm_counts"] == {}


def test_an_arm_with_no_recorded_outcome_is_skipped_not_failed(tmp_path):
    """Off-policy limitation, stated explicitly: unknown != wrong.

    Coverage is per-record, not per-set. The cheap arm is scoreable because
    record 1 logged it, so it stays a candidate for record 2 -- which logged
    only the expensive arm. When the bandit's cost penalty picks the cheap arm
    for record 2 the step must count as unknown and the record must not be
    scored, in either direction.

    Two unknowns, not one, and the count is pinned rather than bounded: record 1
    resolves its logged arm as a 0, cascades to a second arm it never logged,
    and that miss is an unknown too. Only record 1's step 1 is ever scored.
    """
    path = tmp_path / "sparse.json"
    path.write_text(
        json.dumps(
            [
                make_record("q1", "1", [(config.ARM_NAMES[0], 0)]),
                make_record("q2", "2", [(config.ARM_NAMES[-1], 1)]),
            ]
        )
    )
    records = replay.load_replay_records(str(path))
    summary = replay.replay(records, replay.make_offline_router)
    assert summary["unknown_outcomes"] == 2
    # q2's recorded success is never credited, because the arm that earned it
    # is not the arm the router chose.
    assert summary["success_rate"] == 0.0


def test_only_arms_the_records_can_score_are_offered(tmp_path):
    """An arm with no outcome anywhere keeps its full initial optimism, so the
    bandit would pick it every time, find nothing, and discard the record."""
    path = tmp_path / "two_arms.json"
    path.write_text(
        json.dumps(
            [
                make_record("q1", "1", [(config.ARM_NAMES[0], 1)]),
                make_record("q2", "2", [(config.ARM_NAMES[2], 1)]),
            ]
        )
    )
    records = replay.load_replay_records(str(path))
    assert replay.covered_arms(records) == [config.ARM_NAMES[0], config.ARM_NAMES[2]]
    summary = replay.replay(records, replay.make_offline_router)
    assert summary["scoreable_arms"] == 2
    assert set(summary["arm_counts"]) <= {config.ARM_NAMES[0], config.ARM_NAMES[2]}


def test_covered_arms_keeps_config_price_order(tmp_path):
    """estimate_all and the cost penalty both assume ARM_NAMES ordering."""
    records = [{"arm_outcomes": {config.ARM_NAMES[3]: 1, config.ARM_NAMES[1]: 0}}]
    assert replay.covered_arms(records) == [config.ARM_NAMES[1], config.ARM_NAMES[3]]


def test_logged_openrouter_names_are_mapped_onto_bedrock_arms(tmp_path):
    """The logged run used OpenRouter model ids. Unmapped, the name sets are
    disjoint and all 500 records score as unknown."""
    path = tmp_path / "openrouter.json"
    path.write_text(
        json.dumps([make_record("q", "1", [("google/gemini-2.0-flash-001", 1)])])
    )
    records = replay.load_replay_records(str(path))
    assert set(records[0]["arm_outcomes"]) <= set(config.ARM_NAMES)
    assert records[0]["arm_outcomes"][config.ARM_NAMES[1]] == 1


def test_an_unrecognised_logged_model_keeps_its_own_name(tmp_path):
    """Better an honest non-match than a silent mis-credit to a real arm."""
    path = tmp_path / "unknown_model.json"
    path.write_text(json.dumps([make_record("q", "1", [("some/new-model", 1)])]))
    records = replay.load_replay_records(str(path))
    assert set(records[0]["arm_outcomes"]) == {"some/new-model"}


def test_the_real_result_file_maps_onto_real_arms():
    """Guards the disjoint-name-set trap on the actual 500-record log."""
    if not RESULT_FILE.exists():
        pytest.skip(f"{RESULT_FILE} not present")
    records = replay.load_replay_records(str(RESULT_FILE))
    covered = replay.covered_arms(records)
    assert len(covered) >= 6
    assert set(covered) <= set(config.ARM_NAMES)
