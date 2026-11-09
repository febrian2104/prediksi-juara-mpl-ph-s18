from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TeamMatch:
    season: int
    scheduled_at: pd.Timestamp
    opponent: str
    win: float
    game_diff: int
    opponent_elo: float


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
        return {
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
            TeamMatch(season, timestamp, team_b, actual_a, game_diff_a, rating_b)
        )
        self.history.setdefault(team_b, []).append(
            TeamMatch(season, timestamp, team_a, actual_b, -game_diff_a, rating_a)
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
                "match_id": row.match_id,
                "season": int(row.season),
                "stage": row.stage,
                "week": row.week,
                "scheduled_at": row.scheduled_at,
                "team_a_code": row.team_a_code,
                "team_b_code": row.team_b_code,
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
        "roster_features_used": False,
        "roster_feature_reason": "Roster historis bertanggal belum cukup lengkap.",
    }
    (ROOT / "reports" / "feature_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
