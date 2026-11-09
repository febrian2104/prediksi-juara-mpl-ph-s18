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
    artifact = joblib.load(ROOT / "models" / "production_model.joblib")
    report = json.loads((ROOT / "reports" / "model_selection.json").read_text())
    assert best == "elo_baseline"
    assert artifact["model_name"] == best
    assert report["s18_used_for_selection_or_training"] is False
    assert max(report["training_seasons"]) == 17


def test_all_requested_candidate_models_are_evaluated() -> None:
    metrics = pd.read_csv(ROOT / "data" / "predictions" / "walk_forward_metrics.csv")
    assert set(metrics["model"]) == {
        "elo_baseline",
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert (metrics[["accuracy", "log_loss", "brier_score"]].notna()).all().all()
