import json
from pathlib import Path

import pandas as pd

from mpl_ph_predictor.simulation import adjust_best_of

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "predictions"


def test_snapshot_probabilities_are_complete_and_normalized() -> None:
    snapshots = pd.read_csv(PREDICTIONS / "s18_snapshot_probabilities.csv")
    assert snapshots["snapshot_id"].nunique() == 9
    assert set(snapshots.groupby("snapshot_id").size()) == {8}
    assert snapshots[["playoff_probability", "champion_probability"]].ge(0).all().all()
    assert snapshots[["playoff_probability", "champion_probability"]].le(1).all().all()
    playoff_totals = snapshots.groupby("snapshot_id")["playoff_probability"].sum()
    champion_totals = snapshots.groupby("snapshot_id")["champion_probability"].sum()
    assert playoff_totals.between(5.999, 6.001).all()
    assert champion_totals.between(0.999, 1.001).all()
    config = json.loads((ROOT / "config" / "model.json").read_text())
    assert set(snapshots["iterations"]) == {config["simulation"]["iterations"]}


def test_s18_predictions_are_marked_reconstructed() -> None:
    matches = pd.read_csv(PREDICTIONS / "s18_match_predictions.csv")
    summary = pd.read_csv(PREDICTIONS / "s18_champion_summary.csv")
    assert len(matches) == 64
    assert set(matches["prediction_origin"]) == {"reconstructed"}
    assert set(summary["prediction_origin"]) == {"reconstructed"}
    assert set(matches["created_at"]) == {"2026-11-09"}


def test_playoff_bracket_and_actual_champion() -> None:
    playoff = pd.read_csv(PREDICTIONS / "s18_playoff_predictions.csv")
    assert set(playoff["bracket_match_id"]) == {f"M{number}" for number in range(1, 9)}
    final = playoff.set_index("bracket_match_id").loc["M8"]
    assert final["actual_winner_code"] == "RORA"
    assert (final["team_a_score"], final["team_b_score"]) == (2, 4)


def test_longer_series_amplifies_favorite_probability() -> None:
    assert adjust_best_of(0.65, 7) > adjust_best_of(0.65, 5) > 0.65
    assert adjust_best_of(0.35, 7) < adjust_best_of(0.35, 5) < 0.35
