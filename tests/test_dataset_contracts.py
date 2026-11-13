import json
from pathlib import Path

import pandas as pd

from mpl_ph_predictor.data.contracts import conform_table, load_dataset_schema

ROOT = Path(__file__).resolve().parents[1]


def test_reference_columns_exist_in_every_season_and_processed_table() -> None:
    schema = load_dataset_schema()
    for season in range(1, 19):
        folder = ROOT / "data" / (f"mpl-season{season}" if season < 18 else "season18")
        for table, columns in schema["tables"].items():
            frame = pd.read_csv(folder / f"mpl_ph_season_{season:02}_{table}.csv")
            assert frame.columns[: len(columns)].tolist() == columns, (season, table)
    for table, columns in schema["tables"].items():
        frame = pd.read_parquet(ROOT / f"data/processed/canonical/{table}.parquet")
        assert set(columns).issubset(frame.columns)


def test_live_s18_schemas_follow_reference_with_unverified_ids_left_null() -> None:
    schema = load_dataset_schema()
    for table, columns in schema["live_tables"].items():
        frame = pd.read_csv(ROOT / f"data/season18/{table}.csv")
        assert frame.columns[: len(columns)].tolist() == columns
        for column in ("organization_id", "franchise_slot_id", "valid_from", "valid_to"):
            if column in frame:
                assert frame[column].isna().all()
    regular = pd.read_csv(ROOT / "data/season18/schedule_results.csv")
    playoff = pd.read_csv(ROOT / "data/season18/playoff_schedule_results.csv")
    assert len(regular) == 56
    assert len(playoff) == 8


def test_bo2_draw_is_explicit_in_compatible_matches() -> None:
    matches = pd.read_csv(ROOT / "data/mpl-season1/mpl_ph_season_01_matches.csv")
    draws = matches.loc[matches["winner"].eq("draw")]
    assert len(draws) == 1
    assert draws["scheduled_at"].isna().all()
    assert not draws["training_eligible"].any()


def test_reference_numeric_types_preserve_missing_values() -> None:
    result = conform_table(
        pd.DataFrame({"season": [18], "team": ["ONIC Philippines"], "player": ["fixture_only"]}),
        "player_season_stats",
    )
    assert str(result["season"].dtype) == "Int64"
    assert str(result["avg_kills"].dtype) == "Float64"
    assert result["games_played"].isna().all()
    assert result["avg_kills"].isna().all()


def test_official_player_stats_are_postseason_and_do_not_become_s18_features() -> None:
    stats = pd.read_csv(ROOT / "data/season18/mpl_ph_season_18_player_season_stats.csv")
    assert len(stats) == 49
    sources = pd.read_csv(ROOT / "data/raw/source_manifest.csv")
    accessed_at = sources.loc[sources["url"].eq("https://ph-mpl.com/data"), "accessed_at"].iloc[0]
    assert stats["snapshot_date"].eq(accessed_at).all()
    assert not stats["feature_eligible_for_s18"].any()
    assert stats["games_played"].isna().all()  # Not reverse-engineered from rounded averages.
    assert stats["kill_participation_pct"].dropna().between(0, 100).all()
    model = json.loads((ROOT / "config/model.json").read_text())
    assert not set(model["features"]) & {"total_kills", "avg_kills", "avg_kda"}
    absent = stats.loc[stats["player"].eq("Ferdz.")]
    assert absent["total_kills"].isna().all()  # Official '-' is unknown, never zero.


def test_variable_coverage_reports_unavailable_optional_data_honestly() -> None:
    coverage = pd.read_csv(ROOT / "reports/dataset_variable_coverage.csv")
    assert set(coverage["season"]) == set(range(1, 19))
    games = coverage.loc[coverage["table"].eq("games")]
    assert games.query("season == 18 and variable == 'game_number'")["non_null_rows"].sum() > 0
    assert games.loc[games["variable"].isin(["winner", "duration_minutes"])][
        "availability"
    ].eq("unavailable").all()
    stats = coverage.query("season == 18 and table == 'player_season_stats'")
    assert stats.set_index("variable").loc["total_kills", "non_null_rows"] > 0
    assert stats.set_index("variable").loc["games_played", "availability"] == "unavailable"


def test_only_reviewed_aliases_receive_roster_links() -> None:
    stats = pd.read_csv(ROOT / "data/raw/official_s18_player_season_stats.csv")
    reviewed = stats.loc[stats["player_identity_status"].eq("reviewed_season_team_alias")]
    assert set(reviewed["player"]) == {"JMPINKMAN", "Vinnn", "Domeng", "Sensu1"}
    assert reviewed["player_id"].notna().all() and reviewed["role"].notna().all()
    assert reviewed["player_alias_registry_sha256"].str.len().eq(64).all()
    assert not reviewed["feature_eligible_for_training"].any()
    unresolved = stats.loc[stats["player_identity_status"].eq("unresolved_ign_alias")]
    assert unresolved["player_id"].isna().all()
    assert unresolved["role"].isna().all()
