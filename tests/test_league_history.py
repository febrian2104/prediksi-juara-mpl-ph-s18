"""Synthetic match fixtures exercise policies; they are not competition data."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mpl_ph_predictor.data.league_history import (
    annotate_match_participants,
    load_league_history,
)
from mpl_ph_predictor.features import FeatureState, build_match_features, replay_state

ROOT = Path(__file__).resolve().parents[1]


def match(season: int, a: str = "ONIC", b: str = "BREN") -> SimpleNamespace:
    return SimpleNamespace(
        season=season,
        team_a_code=a,
        team_b_code=b,
        team_a_score=2,
        team_b_score=0,
        scheduled_at=pd.Timestamp(f"{2017 + season // 2}-08-01T00:00:00Z"),
        stage="regular_season",
    )


@pytest.mark.parametrize("prior,next_season", [(6, 7), (7, 8), (6, 9)])
def test_pre_franchise_and_boundary_do_not_inherit_rating_or_history(prior, next_season):
    state = FeatureState()
    state.update(match(prior))
    assert state.rating("ONIC") > 1500
    result = state.feature_record("ONIC", "BREN", next_season, pd.Timestamp("2026-01-01T00:00:00Z"))
    assert result["elo_diff"] == 0
    assert result["h2h_diff"] == 0
    assert result["form_3_diff"] == 0
    assert pd.isna(result["history_cutoff_a"])
    assert result["prior_regular_match_win_rate_diff"] is None


def test_franchise_same_code_carryover_is_regressed_but_acquisition_codes_stay_separate():
    state = FeatureState()
    state.update(match(12, "ECHO", "BREN"))
    old_rating = state.rating("ECHO")
    state.begin_season(13)
    assert state.rating("ECHO") == 1500 + 0.72 * (old_rating - 1500)
    assert state.rating("TLPH") == 1500
    assert not state.history.get("TLPH")
    state.update(match(14, "APBR", "BLCK"))
    state.begin_season(15)
    assert state.rating("FLCN") == state.rating("TWIS") == 1500
    assert not state.history.get("FLCN") and not state.history.get("TWIS")
    with pytest.raises(ValueError, match="mundur"):
        state.begin_season(14)


def test_participant_keys_are_season_scoped_and_never_invent_franchise_slots():
    rows = pd.DataFrame(
        {
            "season": [7, 8, 15],
            "team_a_code": ["ONIC"] * 3,
            "team_b_code": ["BREN", "BREN", "FLCN"],
        }
    )
    result = annotate_match_participants(rows)
    assert result.team_a_participant_id.tolist() == ["ph_s07_onic", "ph_s08_onic", "ph_s15_onic"]
    assert result.competition_regime.tolist() == ["pre_franchise", "franchise", "franchise"]
    assert result.franchise_slot_status.tolist() == [
        "not_applicable_pre_franchise",
        "unverified",
        "unverified",
    ]
    assert result[["team_a_franchise_slot_id", "team_b_franchise_slot_id"]].isna().all().all()
    assert "competition_regime" not in rows


def test_collected_participant_catalog_covers_eras_without_claiming_complete_early_seasons():
    participants = pd.read_csv(ROOT / "data/clean/season_participants.csv")
    assert not participants.participant_id.duplicated().any()
    assert set(participants.season) == set(range(1, 19))
    assert (
        participants.loc[participants.season.lt(8), "competition_regime"].eq("pre_franchise").all()
    )
    assert (
        participants.loc[participants.season.le(5), "coverage_status"]
        .eq("partial_observed_participants")
        .all()
    )
    assert participants.franchise_slot_id.isna().all()
    by_season = participants.groupby("season").team_code.agg(set)
    assert {"RSG", "BLCK"}.issubset(by_season[14])
    assert not {"RSG", "BLCK"} & by_season[15]
    assert {"FLCN", "TWIS", "APBR"}.issubset(by_season[15])
    assert "RORA" in by_season[14] and "RORA" not in by_season[13]
    report = json.loads((ROOT / "reports/league_identity_audit.json").read_text())
    assert report["franchise_start_season"] == 8
    assert all(source["status"] == "checksum_verified" for source in report["source_evidence"])
    policy = load_league_history()["rating_policy"]
    assert "never_merge_different_codes" in policy["acquisitions"]


def test_replay_and_training_feature_state_agree_after_franchise_reset():
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    subset = matches.loc[matches.season.le(8)].copy()
    features = build_match_features(subset)
    first_s8 = features.loc[features.season.eq(8)].iloc[0]
    assert first_s8.elo_diff == 0
    assert pd.isna(first_s8.history_cutoff_a)
    state = replay_state(subset, pd.Timestamp(first_s8.scheduled_at))
    record = state.feature_record(
        first_s8.team_a_code, first_s8.team_b_code, 8, pd.Timestamp(first_s8.scheduled_at)
    )
    assert record["elo_diff"] == first_s8.elo_diff
