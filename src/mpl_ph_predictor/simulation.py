from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mpl_ph_predictor.data import compute_standings, load_identity_map
from mpl_ph_predictor.features import FeatureState, load_model_config, replay_state
from mpl_ph_predictor.models import predict_probability

ROOT = Path(__file__).resolve().parents[2]


def _competition_config() -> dict[str, Any]:
    return json.loads((ROOT / "config" / "competition.json").read_text(encoding="utf-8"))


def _series_probability(game_probability: float, best_of: int) -> float:
    needed = best_of // 2 + 1
    return float(
        sum(
            math.comb(best_of, wins)
            * game_probability**wins
            * (1.0 - game_probability) ** (best_of - wins)
            for wins in range(needed, best_of + 1)
        )
    )


def _game_probability_from_bo3(series_probability: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(50):
        midpoint = (low + high) / 2.0
        if _series_probability(midpoint, 3) < series_probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def adjust_best_of(bo3_probability: float, best_of: int) -> float:
    return _series_probability(_game_probability_from_bo3(bo3_probability), best_of)


def matchup_probabilities(
    state: FeatureState,
    teams: list[str],
    model: Any,
    feature_columns: list[str],
    cutoff: pd.Timestamp,
) -> dict[tuple[str, str], float]:
    probabilities: dict[tuple[str, str], float] = {}
    rows = []
    pairs = []
    for team_a in teams:
        for team_b in teams:
            if team_a == team_b:
                continue
            feature = state.feature_record(team_a, team_b, 18, cutoff)
            rows.append({column: feature[column] for column in feature_columns})
            pairs.append((team_a, team_b))
    values = predict_probability(model, pd.DataFrame(rows)[feature_columns])
    for pair, probability in zip(pairs, values, strict=True):
        probabilities[pair] = float(probability)
    return probabilities


def _empty_stats(teams: list[str]) -> dict[str, dict[str, int]]:
    return {team: {"mw": 0, "ml": 0, "gw": 0, "gl": 0} for team in teams}


def _apply_score(
    stats: dict[str, dict[str, int]],
    h2h: dict[tuple[str, str], list[int]],
    team_a: str,
    team_b: str,
    score_a: int,
    score_b: int,
) -> None:
    winner, loser = (team_a, team_b) if score_a > score_b else (team_b, team_a)
    stats[winner]["mw"] += 1
    stats[loser]["ml"] += 1
    stats[team_a]["gw"] += score_a
    stats[team_a]["gl"] += score_b
    stats[team_b]["gw"] += score_b
    stats[team_b]["gl"] += score_a
    h2h.setdefault((team_a, team_b), [0, 0])
    h2h.setdefault((team_b, team_a), [0, 0])
    if winner == team_a:
        h2h[(team_a, team_b)][0] += 1
    else:
        h2h[(team_b, team_a)][0] += 1
    h2h[(team_a, team_b)][1] += score_a - score_b
    h2h[(team_b, team_a)][1] += score_b - score_a


def _rank_teams(
    teams: list[str],
    stats: dict[str, dict[str, int]],
    h2h: dict[tuple[str, str], list[int]],
    rng: np.random.Generator,
) -> list[str]:
    grouped: dict[tuple[int, int], list[str]] = {}
    for team in teams:
        values = stats[team]
        grouped.setdefault((values["mw"], values["gw"] - values["gl"]), []).append(team)
    ranked: list[str] = []
    for main_key in sorted(grouped, reverse=True):
        tied = grouped[main_key]
        tie_values = {}
        for team in tied:
            opponents = [opponent for opponent in tied if opponent != team]
            match_wins = sum(h2h.get((team, opponent), [0, 0])[0] for opponent in opponents)
            game_diff = sum(h2h.get((team, opponent), [0, 0])[1] for opponent in opponents)
            # Residual ties require an official BO1. Its hypothetical result is unknown,
            # so Monte Carlo uses a seeded random draw instead of inventing another rule.
            tie_values[team] = (match_wins, game_diff, rng.random())
        ranked.extend(sorted(tied, key=lambda team: tie_values[team], reverse=True))
    return ranked


def _resolve(reference: str, seeds: list[str], results: dict[str, tuple[str, str]]) -> str:
    if reference.startswith("seed_"):
        return seeds[int(reference.split("_")[1]) - 1]
    outcome, match_id = reference.split("_", maxsplit=1)
    winner, loser = results[match_id]
    return winner if outcome == "winner" else loser


def _simulate_playoffs(
    seeds: list[str],
    probabilities: dict[tuple[str, str], float],
    rng: np.random.Generator,
    bracket: list[dict[str, Any]],
) -> str:
    results: dict[str, tuple[str, str]] = {}
    for match in bracket:
        team_a = _resolve(match["team_a"], seeds, results)
        team_b = _resolve(match["team_b"], seeds, results)
        probability = adjust_best_of(probabilities[(team_a, team_b)], int(match["best_of"]))
        winner, loser = (team_a, team_b) if rng.random() < probability else (team_b, team_a)
        results[match["id"]] = (winner, loser)
    return results["M8"][0]


def _simulate_snapshot(
    regular: pd.DataFrame,
    cutoff: pd.Timestamp,
    probabilities: dict[tuple[str, str], float],
    iterations: int,
    seed: int,
    sweep_probability: float,
) -> pd.DataFrame:
    competition = _competition_config()
    bracket = competition["playoffs"]["matches"]
    teams = sorted(set(regular["team_a_code"]).union(regular["team_b_code"]))
    completed = regular.loc[regular["scheduled_at"].lt(cutoff)]
    remaining = regular.loc[regular["scheduled_at"].ge(cutoff)]
    completed_records = [
        (
            str(row.team_a_code),
            str(row.team_b_code),
            int(row.team_a_score),
            int(row.team_b_score),
        )
        for row in completed.itertuples(index=False)
    ]
    remaining_records = [
        (str(row.team_a_code), str(row.team_b_code)) for row in remaining.itertuples(index=False)
    ]
    playoff_counts = {team: 0 for team in teams}
    champion_counts = {team: 0 for team in teams}
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        stats = _empty_stats(teams)
        h2h: dict[tuple[str, str], list[int]] = {}
        for team_a, team_b, score_a, score_b in completed_records:
            _apply_score(stats, h2h, team_a, team_b, score_a, score_b)
        for team_a, team_b in remaining_records:
            winner_a = rng.random() < probabilities[(team_a, team_b)]
            sweep = rng.random() < sweep_probability
            score_a, score_b = (2, 0 if sweep else 1) if winner_a else (0 if sweep else 1, 2)
            _apply_score(stats, h2h, team_a, team_b, score_a, score_b)
        ranking = _rank_teams(teams, stats, h2h, rng)
        seeds = ranking[:6]
        for team in seeds:
            playoff_counts[team] += 1
        champion_counts[_simulate_playoffs(seeds, probabilities, rng, bracket)] += 1
    return pd.DataFrame(
        {
            "team_code": teams,
            "playoff_probability": [playoff_counts[team] / iterations for team in teams],
            "champion_probability": [champion_counts[team] / iterations for team in teams],
            "iterations": iterations,
        }
    )


def _snapshot_cutoffs(regular: pd.DataFrame) -> list[tuple[str, int, pd.Timestamp]]:
    config = load_model_config()
    snapshots = [
        (
            "preseason",
            0,
            pd.Timestamp(config["simulation"]["preseason_snapshot"]).tz_convert("UTC"),
        )
    ]
    for week, group in regular.groupby("week", sort=True):
        cutoff = group["scheduled_at"].max() + pd.Timedelta(seconds=1)
        snapshots.append((f"week_{int(week)}", int(week), cutoff))
    return snapshots


def build_s18_outputs(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    model: Any,
    artifact: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    model_config = load_model_config()
    identity = load_identity_map().set_index("team_code")["official_name"].to_dict()
    matches = matches.copy()
    matches["scheduled_at"] = pd.to_datetime(matches["scheduled_at"], utc=True)
    features = features.copy()
    features["scheduled_at"] = pd.to_datetime(features["scheduled_at"], utc=True)
    s18_features = features.loc[features["season"].eq(18)].copy()
    probabilities = predict_probability(model, s18_features[artifact["feature_columns"]])
    s18_features["team_a_win_probability"] = probabilities
    s18_features["predicted_winner_code"] = np.where(
        probabilities >= 0.5, s18_features["team_a_code"], s18_features["team_b_code"]
    )
    s18_features["actual_winner_code"] = np.where(
        s18_features["team_a_win"].eq(1), s18_features["team_a_code"], s18_features["team_b_code"]
    )
    s18_features["prediction_correct"] = s18_features["predicted_winner_code"].eq(
        s18_features["actual_winner_code"]
    )
    s18_features["prediction_origin"] = "reconstructed"
    s18_features["created_at"] = "2026-11-09"
    s18_features["model_name"] = artifact["model_name"]
    s18_predictions = s18_features.merge(
        matches[["match_id", "bracket_match_id", "team_a_name", "team_b_name", "best_of"]],
        on="match_id",
        how="left",
    )

    historical_regular = matches.query("season < 18 and stage == 'regular_season'")
    sweep_probability = float(
        historical_regular[["team_a_score", "team_b_score"]].min(axis=1).eq(0).mean()
    )
    regular = matches.query("season == 18 and stage == 'regular_season'").copy()
    snapshot_rows = []
    iterations = int(model_config["simulation"]["iterations"])
    active_teams = sorted(set(regular["team_a_code"]).union(regular["team_b_code"]))
    snapshot_probability_maps: dict[str, dict[tuple[str, str], float]] = {}
    for offset, (snapshot_id, week, cutoff) in enumerate(_snapshot_cutoffs(regular)):
        state = replay_state(matches, cutoff)
        probability_map = matchup_probabilities(
            state, active_teams, model, artifact["feature_columns"], cutoff
        )
        snapshot_probability_maps[snapshot_id] = probability_map
        result = _simulate_snapshot(
            regular,
            cutoff,
            probability_map,
            iterations,
            int(model_config["random_seed"]) + offset,
            sweep_probability,
        )
        result["snapshot_id"] = snapshot_id
        result["week"] = week
        result["cutoff"] = cutoff.isoformat()
        result["prediction_origin"] = "reconstructed"
        snapshot_rows.append(result)
    snapshots = pd.concat(snapshot_rows, ignore_index=True)
    snapshots["official_name"] = snapshots["team_code"].map(identity)

    preplayoff_map = snapshot_probability_maps[f"week_{int(regular['week'].max())}"]
    playoff = matches.query("season == 18 and stage == 'playoffs'").copy()
    playoff["team_a_win_probability"] = playoff.apply(
        lambda row: adjust_best_of(
            preplayoff_map[(row.team_a_code, row.team_b_code)], int(row.best_of)
        ),
        axis=1,
    )
    playoff["predicted_winner_code"] = np.where(
        playoff["team_a_win_probability"].ge(0.5), playoff["team_a_code"], playoff["team_b_code"]
    )
    playoff["actual_winner_code"] = np.where(
        playoff["team_a_score"].gt(playoff["team_b_score"]),
        playoff["team_a_code"],
        playoff["team_b_code"],
    )
    playoff["prediction_correct"] = playoff["predicted_winner_code"].eq(
        playoff["actual_winner_code"]
    )
    playoff["prediction_origin"] = "reconstructed_preplayoff"

    standings = compute_standings(regular)
    standings["official_name"] = standings["team_code"].map(identity)
    preseason = snapshots.loc[snapshots["snapshot_id"].eq("preseason")].sort_values(
        "champion_probability", ascending=False
    )
    preplayoff = snapshots.loc[
        snapshots["snapshot_id"].eq(f"week_{int(regular['week'].max())}")
    ].sort_values("champion_probability", ascending=False)
    summary = pd.DataFrame(
        [
            {
                "prediction_window": "preseason",
                "predicted_champion_code": preseason.iloc[0]["team_code"],
                "predicted_champion_name": preseason.iloc[0]["official_name"],
                "predicted_probability": preseason.iloc[0]["champion_probability"],
                "actual_champion_code": "RORA",
                "actual_champion_name": identity["RORA"],
                "prediction_correct": preseason.iloc[0]["team_code"] == "RORA",
                "prediction_origin": "reconstructed",
            },
            {
                "prediction_window": "preplayoff",
                "predicted_champion_code": preplayoff.iloc[0]["team_code"],
                "predicted_champion_name": preplayoff.iloc[0]["official_name"],
                "predicted_probability": preplayoff.iloc[0]["champion_probability"],
                "actual_champion_code": "RORA",
                "actual_champion_name": identity["RORA"],
                "prediction_correct": preplayoff.iloc[0]["team_code"] == "RORA",
                "prediction_origin": "reconstructed",
            },
        ]
    )
    return {
        "s18_match_predictions": s18_predictions,
        "s18_snapshot_probabilities": snapshots,
        "s18_playoff_predictions": playoff,
        "s18_standings": standings,
        "s18_champion_summary": summary,
    }


def write_s18_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    target = ROOT / "data" / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(target / f"{name}.csv", index=False)
