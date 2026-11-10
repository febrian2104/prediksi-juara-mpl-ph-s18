"""Synthetic fixtures here test timing guards; they are never production data."""

import json

import numpy as np
import pandas as pd
import pytest

from mpl_ph_predictor.archive import read_snapshot, save_snapshot
from mpl_ph_predictor.data.league_history import load_league_history
from mpl_ph_predictor.live import build_live_predictions, evaluate_saved_predictions, project_path
from mpl_ph_predictor.models import EloBaselineModel
from mpl_ph_predictor.models.simulation import _bo1_tie_ranking, _tiebreak_start_order

NOW = pd.Timestamp("2026-11-09T08:00:00Z")


@pytest.fixture
def inputs():
    schedule = pd.DataFrame(
        [
            {
                "match_id": "test_future",
                "season": 18,
                "scheduled_at": "2026-12-01T10:00:00Z",
                "team_a_code": "ONIC",
                "team_b_code": "TLPH",
                "best_of": 3,
                "status": "scheduled",
            }
        ]
    )
    history = pd.DataFrame(
        [
            {
                "match_id": "test_history",
                "season": 17,
                "scheduled_at": "2026-05-01T10:00:00Z",
                "team_a_code": "ONIC",
                "team_b_code": "TLPH",
                "best_of": 3,
                "team_a_score": 2,
                "team_b_score": 0,
                "status": "completed",
                "result_observed_at": "2026-05-01T12:00:00Z",
            }
        ]
    )
    artifact = {
        "model_name": "elo_baseline",
        "feature_columns": ["elo_diff"],
        "training_seasons": [17],
        "trained_at": "2026-11-01T10:00:00Z",
        "training_latest_match_at": "2026-05-01T10:00:00Z",
        "training_data_observed_at": "2026-05-01T12:00:00Z",
        "rating_identity_policy": load_league_history()["rating_policy"],
    }
    rules = {
        "competition": "MPL Philippines",
        "season": 18,
        "participants": ["ONIC", "TLPH"],
        "verification": {
            "status": "verified",
            "season": 18,
            "source_url": "https://ph-mpl.com/rules",
            "verified_at": "2026-11-01T00:00:00Z",
        },
    }
    return schedule, history, artifact, rules


def predict(inputs):
    schedule, history, artifact, rules = inputs
    return build_live_predictions(schedule, history, EloBaselineModel(), artifact, rules, now=NOW)


def test_live_cannot_be_backdated_or_include_known_scores(inputs) -> None:
    schedule, _, _, _ = inputs
    schedule.loc[0, "scheduled_at"] = NOW.isoformat()
    with pytest.raises(ValueError, match="belum dimulai"):
        predict(inputs)
    schedule.loc[0, "scheduled_at"] = "2026-12-01T10:00:00Z"
    schedule.loc[0, "team_a_score"] = 2
    with pytest.raises(ValueError, match="sudah diketahui"):
        predict(inputs)


def test_future_observed_results_do_not_change_forecast(inputs) -> None:
    baseline = predict(inputs)
    schedule, history, artifact, rules = inputs
    poisoned = history.copy()
    poisoned["match_id"] = "future_observation"
    poisoned["team_a_score"], poisoned["team_b_score"] = 0, 2
    poisoned["result_observed_at"] = "2026-12-02T00:00:00Z"
    extended = pd.concat([history, poisoned], ignore_index=True)
    result = predict((schedule, extended, artifact, rules))
    pd.testing.assert_frame_equal(baseline, result)


def test_rules_from_another_season_are_rejected(inputs) -> None:
    inputs[3]["season"] = 19
    with pytest.raises(ValueError, match="musim yang sama"):
        predict(inputs)


def test_model_from_the_future_is_rejected(inputs) -> None:
    inputs[2]["trained_at"] = "2026-12-02T00:00:00Z"
    with pytest.raises(ValueError, match="Provenance waktu model"):
        predict(inputs)


def test_live_rejects_artifact_trained_with_another_identity_policy(inputs) -> None:
    inputs[2]["rating_identity_policy"] = {"version": "old_policy"}
    with pytest.raises(ValueError, match="Kebijakan identitas"):
        predict(inputs)


def test_exclusive_snapshot_survives_evaluation_and_rejects_overwrite(tmp_path, inputs) -> None:
    predictions = predict(inputs)
    path = save_snapshot(
        {"predictions.csv": predictions.to_csv(index=False).encode()},
        kind="saved_pre_match",
        metadata={},
        directory=tmp_path,
        key="week_01",
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_snapshot(
            {"predictions.csv": b"changed"},
            kind="saved_pre_match",
            metadata={},
            directory=tmp_path,
            key="week_01",
        )
    results = inputs[0].copy()
    results["team_a_score"], results["team_b_score"] = 2, 1
    results["status"] = "completed"
    results["result_observed_at"] = "2026-12-01T12:00:00Z"
    evaluation = evaluate_saved_predictions(
        predictions, results, now=pd.Timestamp("2026-12-02T00:00:00Z")
    )
    assert evaluation.loc[0, "evaluation_status"] == "evaluated"
    assert bool(evaluation.loc[0, "prediction_correct"])
    assert path.read_bytes() == original
    assert read_snapshot(path)[0]["kind"] == "saved_pre_match"


def test_modified_snapshot_is_detected(tmp_path) -> None:
    path = save_snapshot(
        {"test.csv": b"a,b\n1,2\n"}, kind="reconstruction_archive", metadata={}, directory=tmp_path
    )
    envelope = json.loads(path.read_text())
    envelope["payload"]["metadata"]["changed"] = True
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="Integritas"):
        read_snapshot(path)


def test_snapshot_keys_and_paths_stay_inside_project(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_snapshot(
            {"test.csv": b"a"}, kind="test", metadata={}, directory=tmp_path, key="../escape"
        )
    with pytest.raises(ValueError, match="folder proyek"):
        project_path("../outside.csv")


@pytest.mark.parametrize("count", range(2, 9))
def test_bo1_brackets_produce_one_complete_ranking(count) -> None:
    teams = [f"test_{i}" for i in range(count)]
    ranking = _bo1_tie_ranking(teams, {}, np.random.default_rng(42))
    assert len(ranking) == count
    assert set(ranking) == set(teams)


def test_bo1_tiebreak_uses_team_probability() -> None:
    probabilities = {("a", "b"): 1.0}
    assert _bo1_tie_ranking(["a", "b"], probabilities, np.random.default_rng(2)) == ["a", "b"]


def test_tiebreak_duration_order_is_exact_when_complete_and_strict_rejects_unknowns():
    rng = np.random.default_rng(42)
    teams = ["fixture_a", "fixture_b", "fixture_c"]
    durations = {"fixture_a": 50.0, "fixture_b": 20.0, "fixture_c": 30.0}
    assert _tiebreak_start_order(teams, durations, rng, strict=True) == [
        "fixture_b", "fixture_c", "fixture_a",
    ]
    for incomplete in (None, {}, {"fixture_a": float("nan")},
                       {"fixture_a": -1.0}, {"fixture_a": None}):
        with pytest.raises(ValueError, match="durasi kemenangan"):
            _tiebreak_start_order(teams, incomplete, rng, strict=True)
    assert set(_tiebreak_start_order(teams, None, rng)) == set(teams)


def test_equal_victory_durations_draw_only_within_the_equal_duration_group():
    teams = ["fixture_a", "fixture_b", "fixture_c", "fixture_d"]
    durations = dict(zip(teams, [10.0, 20.0, 20.0, 30.0], strict=True))
    for seed in range(5):
        order = _tiebreak_start_order(teams, durations, np.random.default_rng(seed), strict=True)
        assert order[0] == "fixture_a" and order[-1] == "fixture_d"
        assert set(order[1:3]) == {"fixture_b", "fixture_c"}
