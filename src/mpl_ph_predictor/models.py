from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mpl_ph_predictor.features import load_model_config

ROOT = Path(__file__).resolve().parents[2]


class EloBaselineModel:
    """Scikit-learn-like probability interface for the evaluated Elo baseline."""

    def __init__(self, scale: float = 400.0) -> None:
        self.scale = scale

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        differences = x["elo_diff"].to_numpy(dtype=float)
        probability = 1.0 / (1.0 + np.power(10.0, -differences / self.scale))
        return np.column_stack([1.0 - probability, probability])


def candidate_models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.7, max_iter=2000, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=8,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=2,
                        learning_rate=0.035,
                        subsample=0.8,
                        colsample_bytree=0.85,
                        reg_lambda=3.0,
                        min_child_weight=6,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
    }


def _augment_symmetric(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    inverted = -x.copy()
    return (
        pd.concat([x, inverted], ignore_index=True),
        pd.concat([y.reset_index(drop=True), (1 - y).reset_index(drop=True)], ignore_index=True),
    )


def predict_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    direct = model.predict_proba(x)[:, 1]
    inverse = 1.0 - model.predict_proba(-x)[:, 1]
    return np.clip((direct + inverse) / 2.0, 1e-6, 1.0 - 1e-6)


def _calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities >= left) & (
            (probabilities <= right) if right == 1.0 else (probabilities < right)
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(y[mask].mean()) - float(probabilities[mask].mean())
            )
    return error if total else float("nan")


def _metrics(y: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    actual = y.to_numpy(dtype=int)
    return {
        "accuracy": float(accuracy_score(actual, probabilities >= 0.5)),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(actual, probabilities)),
        "expected_calibration_error": _calibration_error(actual, probabilities),
    }


def walk_forward_compare(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_model_config()
    feature_columns = config["features"]
    first_test = int(config["walk_forward"]["first_test_season"])
    minimum_rows = int(config["walk_forward"]["minimum_training_rows"])
    seed = int(config["random_seed"])
    historical = features.loc[features["season"].lt(18)].copy()
    predictions: list[dict[str, Any]] = []
    test_seasons = sorted(
        int(value) for value in historical["season"].unique() if value >= first_test
    )
    for test_season in test_seasons:
        train = historical.loc[historical["season"].lt(test_season)]
        test = historical.loc[historical["season"].eq(test_season)]
        if len(train) < minimum_rows or test.empty:
            continue
        x_train, y_train = _augment_symmetric(train[feature_columns], train["team_a_win"])
        for name, model in candidate_models(seed).items():
            model.fit(x_train, y_train)
            probabilities = predict_probability(model, test[feature_columns])
            for row, probability in zip(test.itertuples(index=False), probabilities, strict=True):
                predictions.append(
                    {
                        "match_id": row.match_id,
                        "season": test_season,
                        "model": name,
                        "team_a_win": int(row.team_a_win),
                        "probability": float(probability),
                        "prediction_origin": "walk_forward_out_of_sample",
                    }
                )
        for row in test.itertuples(index=False):
            predictions.append(
                {
                    "match_id": row.match_id,
                    "season": test_season,
                    "model": "elo_baseline",
                    "team_a_win": int(row.team_a_win),
                    "probability": float(row.elo_probability),
                    "prediction_origin": "walk_forward_out_of_sample",
                }
            )
    prediction_frame = pd.DataFrame(predictions)
    metric_rows = []
    for model, group in prediction_frame.groupby("model"):
        metric_rows.append(
            {
                "model": model,
                "rows": len(group),
                **_metrics(group["team_a_win"], group["probability"].to_numpy()),
            }
        )
    metrics = (
        pd.DataFrame(metric_rows).sort_values(["log_loss", "brier_score"]).reset_index(drop=True)
    )
    return metrics, prediction_frame


def fit_production_model(
    features: pd.DataFrame, metrics: pd.DataFrame
) -> tuple[str, Any, dict[str, Any]]:
    config = load_model_config()
    selected = str(metrics.sort_values(["log_loss", "brier_score"]).iloc[0]["model"])
    training = features.loc[features["season"].lt(18)]
    if selected == "elo_baseline":
        model = EloBaselineModel(scale=float(config["elo"]["scale"]))
        production_features = ["elo_diff"]
    else:
        x_train, y_train = _augment_symmetric(training[config["features"]], training["team_a_win"])
        model = candidate_models(int(config["random_seed"]))[selected]
        model.fit(x_train, y_train)
        production_features = config["features"]
    artifact = {
        "model_name": selected,
        "feature_columns": production_features,
        "training_seasons": sorted(int(value) for value in training["season"].unique()),
        "training_rows_original": int(len(training)),
        "selection_method": "lowest walk-forward log loss, then Brier score",
        "s18_used_for_selection_or_training": False,
    }
    return selected, model, artifact


def write_model_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    model: Any,
    artifact: dict[str, Any],
) -> None:
    predictions_dir = ROOT / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(predictions_dir / "walk_forward_metrics.csv", index=False)
    predictions.to_csv(predictions_dir / "walk_forward_predictions.csv", index=False)
    joblib.dump({"model": model, **artifact}, ROOT / "models" / "production_model.joblib")
    (ROOT / "reports" / "model_selection.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
