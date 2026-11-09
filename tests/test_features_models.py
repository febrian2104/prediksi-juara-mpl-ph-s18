import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_features_do_not_use_future_matches() -> None:
    features = pd.read_csv(ROOT / "data" / "features" / "match_features.csv")
    assert not features["history_after_match"].astype(bool).any()
    assert not features["match_id"].duplicated().any()


def test_walk_forward_uses_earlier_seasons_and_selects_best_model() -> None:
    metrics = pd.read_csv(ROOT / "data" / "predictions" / "walk_forward_metrics.csv")
    best = metrics.sort_values(["log_loss", "brier_score"]).iloc[0]["model"]
    artifact = joblib.load(ROOT / "artifacts" / "final_match_model.joblib")
    report = json.loads((ROOT / "reports" / "model_selection.json").read_text())
    assert artifact["model_name"] == best
    assert report["s18_used_for_selection_or_training"] is False
    assert max(report["training_seasons"]) == 17
    assert pd.Timestamp(report["trained_at"]) >= pd.Timestamp(report["training_data_observed_at"])


def test_all_requested_candidate_models_are_evaluated() -> None:
    metrics = pd.read_csv(ROOT / "data" / "predictions" / "walk_forward_metrics.csv")
    assert set(metrics["model"]) == {
        "elo_baseline",
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert (metrics[["accuracy", "log_loss", "brier_score"]].notna()).all().all()


def test_reference_feature_names_keep_current_and_prior_seasons_distinct() -> None:
    from types import SimpleNamespace

    from mpl_ph_predictor.features import FeatureState

    state = FeatureState()
    state.update(
        SimpleNamespace(
            season=17,
            scheduled_at=pd.Timestamp("2026-05-01T00:00:00Z"),
            team_a_code="fixture_a",
            team_b_code="fixture_b",
            team_a_score=2,
            team_b_score=0,
            stage="regular_season",
        )
    )
    record = state.feature_record(
        "fixture_a", "fixture_b", 18, pd.Timestamp("2026-08-01T00:00:00Z")
    )
    assert record["current_matches_diff"] == 0
    assert record["current_form_3_diff"] is None
    assert record["prior_regular_match_win_rate_diff"] == 1
    assert record["elo_rating_diff"] == record["elo_diff"]
    features = pd.read_csv(ROOT / "data/features/match_features.csv")
    assert features["completion_date"].isna().all()
    assert features["team_a_franchise_slot_id"].isna().all()
