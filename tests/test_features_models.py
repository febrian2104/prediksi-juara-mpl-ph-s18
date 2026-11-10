import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

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


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost"])
def test_candidate_models_can_fit_a_small_real_historical_sample(name: str) -> None:
    import numpy as np

    from mpl_ph_predictor.features import load_model_config
    from mpl_ph_predictor.models.evaluation import (
        _augment_symmetric,
        candidate_models,
        predict_probability,
    )

    config = load_model_config()
    features = pd.read_csv(ROOT / "data/features/match_features.csv")
    sample = features.loc[features.season.lt(18)].head(64)
    x, y = _augment_symmetric(sample[config["features"]], sample.team_a_win)
    model = candidate_models(config["random_seed"])[name]
    model.fit(x, y)
    probability = predict_probability(model, x.head(8))
    reverse = predict_probability(model, -x.head(8))
    assert np.isfinite(probability).all()
    assert ((probability > 0) & (probability < 1)).all()
    np.testing.assert_allclose(probability + reverse, 1.0, atol=1e-10)


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


def test_linux_x86_runtime_uses_one_official_cpu_distribution() -> None:
    import platform
    from importlib.metadata import PackageNotFoundError, version

    if platform.system() == "Linux" and platform.machine() == "x86_64":
        assert version("xgboost-cpu") == "3.1.1"
        with pytest.raises(PackageNotFoundError):
            version("xgboost")
