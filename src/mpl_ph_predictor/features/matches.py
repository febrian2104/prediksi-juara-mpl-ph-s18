from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TeamMatch:
    season: int
    scheduled_at: pd.Timestamp
    opponent: str
    win: float
    game_diff: int
    opponent_elo: float
    stage: str = "unknown"
    games_for: int = 0
    games_against: int = 0


@dataclass
class FeatureState:
    initial_rating: float = 1500.0
    k_factor: float = 28.0
    scale: float = 400.0
    season_carryover: float = 0.72
    ratings: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[TeamMatch]] = field(default_factory=dict)
    current_season: int | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureState:
        return cls(
            initial_rating=float(config["initial_rating"]),
            k_factor=float(config["k_factor"]),
            scale=float(config["scale"]),
            season_carryover=float(config["season_carryover"]),
        )

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def begin_season(self, season: int) -> None:
        if self.current_season == season:
            return
        if self.current_season is not None:
            for team, rating in list(self.ratings.items()):
                self.ratings[team] = self.initial_rating + self.season_carryover * (
                    rating - self.initial_rating
                )
        self.current_season = season

    def expected(self, team_a: str, team_b: str) -> float:
        return 1.0 / (1.0 + 10.0 ** ((self.rating(team_b) - self.rating(team_a)) / self.scale))

    def _recent(self, team: str, n: int) -> list[TeamMatch]:
        return self.history.get(team, [])[-n:]

    def _form(self, team: str, n: int) -> float:
        recent = self._recent(team, n)
        return float(np.mean([row.win for row in recent])) if recent else 0.5

    def _game_diff(self, team: str, n: int) -> float:
        recent = self._recent(team, n)
        return float(np.mean([row.game_diff for row in recent])) if recent else 0.0

    def _sos(self, team: str, n: int) -> float:
        recent = self._recent(team, n)
        return (
            float(np.mean([row.opponent_elo for row in recent])) if recent else self.initial_rating
        )

    def _h2h(self, team: str, opponent: str) -> float:
        meetings = [row for row in self.history.get(team, []) if row.opponent == opponent][-6:]
        return float(np.mean([row.win for row in meetings])) if meetings else 0.5

    def feature_record(
        self, team_a: str, team_b: str, season: int, scheduled_at: pd.Timestamp
    ) -> dict[str, Any]:
        self.begin_season(season)
        history_a = self.history.get(team_a, [])
        history_b = self.history.get(team_b, [])
        current_a = sum(row.season == season for row in history_a)
        current_b = sum(row.season == season for row in history_b)
        cutoff_a = history_a[-1].scheduled_at if history_a else pd.NaT
        cutoff_b = history_b[-1].scheduled_at if history_b else pd.NaT
        result = {
            "elo_diff": self.rating(team_a) - self.rating(team_b),
            "elo_probability": self.expected(team_a, team_b),
            "form_3_diff": self._form(team_a, 3) - self._form(team_b, 3),
            "form_5_diff": self._form(team_a, 5) - self._form(team_b, 5),
            "game_diff_5_diff": self._game_diff(team_a, 5) - self._game_diff(team_b, 5),
            "sos_5_diff": self._sos(team_a, 5) - self._sos(team_b, 5),
            "h2h_diff": self._h2h(team_a, team_b) - self._h2h(team_b, team_a),
            "season_matches_diff": current_a - current_b,
            "history_cutoff_a": cutoff_a,
            "history_cutoff_b": cutoff_b,
            "feature_timestamp": scheduled_at,
        }
        result.update(self.reference_features(team_a, team_b, season, scheduled_at))
        return result

    def reference_features(
        self, team_a: str, team_b: str, season: int, timestamp: pd.Timestamp
    ) -> dict[str, Any]:
        def summary(rows: list[TeamMatch]) -> dict[str, float | None]:
            total_games = sum(row.games_for + row.games_against for row in rows)
            return {
                "match_win_rate": float(np.mean([row.win for row in rows])) if rows else None,
                "game_win_rate": sum(row.games_for for row in rows) / total_games
                if total_games
                else None,
                "game_diff_per_match": float(np.mean([row.game_diff for row in rows]))
                if rows
                else None,
            }

        def team_state(team: str) -> dict[str, Any]:
            history = self.history.get(team, [])
            current = [row for row in history if row.season == season]
            prior = [
                row for row in history if row.season == season - 1 and row.stage == "regular_season"
            ]
            rolling = [row for row in history if season - 3 <= row.season < season]
            values = {"current_matches": len(current)}
            for prefix, rows in (
                ("current", current),
                ("prior_regular", prior),
                ("prior_3_season", rolling),
            ):
                values.update({f"{prefix}_{name}": value for name, value in summary(rows).items()})
            for count in (3, 5):
                values[f"current_form_{count}"] = (
                    float(np.mean([row.win for row in current[-count:]])) if current else None
                )
            values["current_sos_elo_avg"] = (
                float(np.mean([row.opponent_elo for row in current])) if current else None
            )
            values["rest_days"] = (
                (timestamp - history[-1].scheduled_at).total_seconds() / 86400 if history else None
            )
            return values

        state_a, state_b = team_state(team_a), team_state(team_b)
        result = {}
        for name in state_a:
            a, b = state_a[name], state_b[name]
            result[f"{name}_diff"] = float(a - b) if a is not None and b is not None else None
        result.update(
            {
                "elo_rating_a": self.rating(team_a),
                "elo_rating_b": self.rating(team_b),
                "elo_rating_diff": self.rating(team_a) - self.rating(team_b),
                "elo_expected_team_a": self.expected(team_a, team_b),
                "latest_history_date_a": self.history[team_a][-1].scheduled_at
                if self.history.get(team_a)
                else pd.NaT,
                "latest_history_date_b": self.history[team_b][-1].scheduled_at
                if self.history.get(team_b)
                else pd.NaT,
                "history_date_basis": "scheduled_time_proxy_completion_time_unverified",
            }
        )
        return result

    def update(self, row: Any) -> None:
        season = int(row.season)
        self.begin_season(season)
        team_a = str(row.team_a_code)
        team_b = str(row.team_b_code)
        rating_a = self.rating(team_a)
        rating_b = self.rating(team_b)
        expected_a = self.expected(team_a, team_b)
        actual_a = float(int(row.team_a_score) > int(row.team_b_score))
        actual_b = 1.0 - actual_a
        self.ratings[team_a] = rating_a + self.k_factor * (actual_a - expected_a)
        self.ratings[team_b] = rating_b + self.k_factor * (actual_b - (1.0 - expected_a))
        timestamp = pd.Timestamp(row.scheduled_at)
        game_diff_a = int(row.team_a_score) - int(row.team_b_score)
        self.history.setdefault(team_a, []).append(
            TeamMatch(
                season,
                timestamp,
                team_b,
                actual_a,
                game_diff_a,
                rating_b,
                getattr(row, "stage", "unknown"),
                int(row.team_a_score),
                int(row.team_b_score),
            )
        )
        self.history.setdefault(team_b, []).append(
            TeamMatch(
                season,
                timestamp,
                team_a,
                actual_b,
                -game_diff_a,
                rating_a,
                getattr(row, "stage", "unknown"),
                int(row.team_b_score),
                int(row.team_a_score),
            )
        )


def load_model_config() -> dict[str, Any]:
    return json.loads((ROOT / "config" / "model.json").read_text(encoding="utf-8"))


def build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    config = load_model_config()
    state = FeatureState.from_config(config["elo"])
    rows: list[dict[str, Any]] = []
    ordered = matches.copy()
    ordered["scheduled_at"] = pd.to_datetime(ordered["scheduled_at"], utc=True)
    ordered = ordered.sort_values(["season", "scheduled_at", "match_id"])
    for row in ordered.itertuples(index=False):
        record = state.feature_record(
            str(row.team_a_code), str(row.team_b_code), int(row.season), row.scheduled_at
        )
        rows.append(
            {
                "feature_version": "ph_2_reference_columns",
                "match_id": row.match_id,
                "season": int(row.season),
                "stage": row.stage,
                "week": row.week,
                "scheduled_at": row.scheduled_at,
                "team_a_code": row.team_a_code,
                "team_b_code": row.team_b_code,
                "team_a_id": getattr(row, "team_a_id", pd.NA),
                "team_b_id": getattr(row, "team_b_id", pd.NA),
                "team_a_franchise_slot_id": pd.NA,
                "team_b_franchise_slot_id": pd.NA,
                "completion_date": pd.NaT,
                "team_a_score": int(row.team_a_score),
                "team_b_score": int(row.team_b_score),
                "team_a_win": int(row.team_a_score > row.team_b_score),
                **record,
            }
        )
        state.update(row)
    features = pd.DataFrame(rows)
    features["history_after_match"] = (
        features["history_cutoff_a"].ge(features["feature_timestamp"])
        | features["history_cutoff_b"].ge(features["feature_timestamp"])
    ).fillna(False)
    return features


def replay_state(matches: pd.DataFrame, cutoff: pd.Timestamp) -> FeatureState:
    config = load_model_config()
    state = FeatureState.from_config(config["elo"])
    ordered = matches.copy()
    ordered["scheduled_at"] = pd.to_datetime(ordered["scheduled_at"], utc=True)
    for row in (
        ordered.loc[ordered["scheduled_at"].lt(cutoff)]
        .sort_values(["season", "scheduled_at", "match_id"])
        .itertuples(index=False)
    ):
        state.update(row)
    return state


def write_features(features: pd.DataFrame) -> dict[str, Any]:
    target = ROOT / "data" / "features"
    target.mkdir(parents=True, exist_ok=True)
    features.to_csv(target / "match_features.csv", index=False)
    report = {
        "rows": int(len(features)),
        "seasons": sorted(int(value) for value in features["season"].unique()),
        "duplicate_match_ids": int(features["match_id"].duplicated().sum()),
        "history_after_match_count": int(features["history_after_match"].sum()),
        "feature_columns": load_model_config()["features"],
        "reference_compatible_columns": [
            column
            for column in features
            if column.startswith(
                ("elo_rating", "elo_expected", "current_", "prior_", "latest_history", "rest_days")
            )
        ],
        "completion_time_status": "unverified_kept_null",
        "roster_features_used": False,
        "roster_feature_reason": "Roster historis bertanggal belum cukup lengkap.",
    }
    (ROOT / "reports" / "feature_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
