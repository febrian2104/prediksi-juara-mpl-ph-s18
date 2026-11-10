import json
from pathlib import Path

import pandas as pd
import pytest

from mpl_ph_predictor.archive import sha256
from mpl_ph_predictor.data.identity import apply_team_name_history
from mpl_ph_predictor.data.season_evidence import (
    collect_season_evidence,
    parse_archive_standings,
    parse_archive_statistics,
    reconcile_standings,
)

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = {
    "url": "https://ph-mpl.com/s8/data",
    "retrieved_at": "2026-11-10T01:00:00Z",
    "information_cutoff": "2026-11-09",
    "snapshot_path": "fixture_only.html",
}


def stats_fixture(kills: str = "5", kp: str = "50%") -> bytes:
    headers = [
        "Player",
        "Team",
        "Total Kills",
        "AVG Kills",
        "Total Deaths",
        "AVG Deaths",
        "Total Assists",
        "AVG Assists",
        "KDA Ratio",
        "Kill Participation",
    ]
    values = ["fixture_player", "TLPH", kills, "2.50", "-", "-", "8", "4.00", "-", kp]
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead><tbody><tr>"
        + "".join(f"<td>{v}</td>" for v in values)
        + "</tr></tbody></table>"
    ).encode()


def test_archive_statistics_preserve_unknowns_and_never_infer_games_or_roster() -> None:
    result = parse_archive_statistics(stats_fixture(), 8, PROVENANCE)
    row = result.iloc[0]
    assert row.team_code == "ECHO" and row.raw_team_code == "TLPH"
    assert row.total_kills == 5 and row.kill_participation_pct == 50
    assert result[["games_played", "player_id", "role", "total_deaths"]].isna().all().all()
    assert not row.feature_eligible_for_training and not row.feature_eligible_for_s18
    assert row.team_assignment_status == "archive_label_not_effective_dated_roster"


@pytest.mark.parametrize("kills,kp", [("2.5", "50%"), ("5", "101%"), ("NaN", "50%")])
def test_invalid_archive_statistics_are_rejected(kills: str, kp: str) -> None:
    with pytest.raises(ValueError):
        parse_archive_statistics(stats_fixture(kills, kp), 8, PROVENANCE)


def test_grouped_average_headers_do_not_become_total_counts_or_kill_participation() -> None:
    headers = [
        "PLAYER",
        "TEAM",
        "AVERAGE",
        "TOTAL",
        "KILLS",
        "DEATHS",
        "ASSISTS",
        "KDA",
        "DAMAGE/GOLD",
        "TEAMFIGHT",
        "HERO USED",
    ]
    values = ["fixture_player", "TLPH", "1.75", "1.79", "5.77", "4.20", "0.00", "70.92%", "11"]
    html = (
        "<table><thead><tr>"
        + "".join(f"<th>{value}</th>" for value in headers)
        + "</tr></thead><tbody><tr>"
        + "".join(f"<td>{value}</td>" for value in values)
        + "</tr></tbody></table>"
    ).encode()
    result = parse_archive_statistics(html, 13, PROVENANCE)
    assert result.iloc[0].avg_kills == 1.75 and result.iloc[0].avg_assists == 5.77
    assert result.iloc[0].teamfight_pct == 70.92 and result.iloc[0].heroes_used_count == 11
    unknown = ["total_kills", "total_deaths", "total_assists", "kill_participation_pct"]
    assert result[unknown].isna().all().all()
    invalid = parse_archive_statistics(html.replace(b"70.92%", b"373.13%"), 12, PROVENANCE)
    assert invalid.teamfight_pct.isna().all()
    assert invalid.iloc[0].teamfight_pct_source_value == "373.13%"
    assert invalid.iloc[0].row_quality_issues == "teamfight_pct_out_of_range"
    assert invalid.iloc[0].avg_kills == 1.75  # Unrelated source values remain unchanged.


def test_dated_s13_names_do_not_invent_an_effective_hour_or_change_rating_keys() -> None:
    fixture = pd.DataFrame(
        {
            "season": [13, 13, 18],
            "scheduled_at": [
                "2024-05-05T08:00:00Z",
                "2024-05-26T10:00:00Z",
                "2026-08-21T08:00:00Z",
            ],
            "team_a_code": ["TLPH"] * 3,
            "team_b_code": ["APBR"] * 3,
            "team_a_name": ["Team Liquid PH"] * 3,
            "team_b_name": ["AP BREN"] * 3,
        }
    )
    result = apply_team_name_history(fixture)
    assert result.team_a_name.tolist() == ["ECHO", "Team Liquid ECHO PH", "Team Liquid PH"]
    assert result.team_a_code.equals(fixture.team_a_code)
    fixture.loc[0, "scheduled_at"] = "2024-05-21T12:00:00Z"
    with pytest.raises(ValueError, match="jam perlu audit"):
        apply_team_name_history(fixture)


def test_reconciliation_compares_wl_not_s18_points_and_does_not_mutate_matches() -> None:
    matches = pd.DataFrame(
        {
            "season": [8],
            "stage": ["regular_season"],
            "team_a_code": ["ECHO"],
            "team_b_code": ["BREN"],
            "team_a_score": [2],
            "team_b_score": [1],
        }
    )
    original = matches.copy(deep=True)
    html = b"""<table><thead><tr><th>RANK</th><th>TEAM</th><th>MATCH W-L</th>
    <th>GAMES W-L</th><th>POINTS</th></tr></thead><tbody>
    <tr><td>1</td><td>TLPH</td><td>1 - 0</td><td>2 - 1</td><td>2</td></tr>
    <tr><td>2</td><td>BREN</td><td>0 - 1</td><td>1 - 2</td><td>1</td></tr>
    </tbody></table>"""
    standings = parse_archive_standings(html, 8, PROVENANCE)
    assert reconcile_standings(matches, standings, 8)["status"] == "regular_wl_totals_reconciled"
    standings.loc[0, "game_wins"] = 5
    report = reconcile_standings(matches, standings, 8)
    assert report["status"] == "source_conflict_or_scope_mismatch"
    assert report["difference_count"] == 1
    pd.testing.assert_frame_equal(matches, original)


@pytest.mark.parametrize("scope", ["season", "mixed"])
def test_season_and_mixed_scopes_do_not_masquerade_as_regular_standings(scope: str) -> None:
    matches = pd.DataFrame(
        {
            "season": [10, 10],
            "stage": ["regular_season", "playoffs"],
            "team_a_code": ["FIXTURE_A"] * 2,
            "team_b_code": ["FIXTURE_B"] * 2,
            "team_a_score": [2, 3],
            "team_b_score": [0, 1],
        }
    )
    original = matches.copy(deep=True)
    observations = pd.DataFrame(
        {
            "team_code": ["FIXTURE_A", "FIXTURE_B"],
            "match_wins": [2 if scope == "season" else 1, 0],
            "match_losses": [0, 2 if scope == "season" else 1],
            "game_wins": [5, 1],
            "game_losses": [1, 5],
        }
    )
    report = reconcile_standings(matches, observations, 10)
    expected = (
        "season_wl_totals_reconciled" if scope == "season" else ("mixed_scope_wl_totals_reconciled")
    )
    assert report["status"] == expected
    assert not report["game_wl_totals_reconciled"]
    assert report["residual_difference_count"] == 0
    assert "regular_season" not in report["exact_matching_scopes"]
    observations.loc[0, "game_wins"] += 1
    conflict = reconcile_standings(matches, observations, 10)
    assert conflict["status"] == "source_conflict_or_scope_mismatch"
    assert conflict["residual_difference_count"] == 1
    pd.testing.assert_frame_equal(matches, original)


def test_real_s10_archive_matches_all_stages_but_s8_has_an_unresolved_residual() -> None:
    audits = json.loads((ROOT / "reports/season_reconciliation.json").read_text())
    by_season = {row["season"]: row for row in audits}
    assert by_season[10]["status"] == "season_wl_totals_reconciled"
    assert by_season[10]["archive_wl_scope"] == "all_collected_matches"
    assert by_season[10]["residual_difference_count"] == 0
    assert by_season[8]["status"] == "source_conflict_or_scope_mismatch"
    assert by_season[8]["residual_difference_count"] > 0


def test_fnatic_competition_start_date_is_not_the_announcement_date() -> None:
    fixture = pd.DataFrame(
        {
            "season": [13, 13, 13],
            "scheduled_at": [
                "2024-05-21T10:00:00Z",
                "2024-05-22T08:00:00Z",
                "2024-05-23T08:00:00Z",
            ],
            "team_a_code": ["ONIC"] * 3,
            "team_b_code": ["FIXTURE_B"] * 3,
            "team_a_name": ["ONIC Philippines"] * 3,
            "team_b_name": ["Fixture"] * 3,
        }
    )
    result = apply_team_name_history(fixture)
    assert result.team_a_name.tolist() == ["ONIC Philippines", "Fnatic ONIC PH", "Fnatic ONIC PH"]
    assert result.team_a_code.equals(fixture.team_a_code)


@pytest.mark.parametrize(
    "code,before,after",
    [
        ("ONIC", "ONIC Philippines", "Fnatic ONIC PH"),
        ("APBR", "AP.Bren", "Falcons APBren"),
    ],
)
def test_s13_additional_names_are_date_aware_and_do_not_duplicate_team_ids(
    code: str,
    before: str,
    after: str,
) -> None:
    matches = pd.read_csv(ROOT / "data/clean/matches.csv").query("season == 13")
    for side in ("a", "b"):
        rows = matches.loc[matches[f"team_{side}_code"].eq(code)]
        assert set(rows.loc[rows.stage.eq("regular_season"), f"team_{side}_name"]) <= {before}
        assert set(rows.loc[rows.stage.eq("playoffs"), f"team_{side}_name"]) <= {after}
    teams = pd.read_csv(ROOT / "data/mpl-season13/mpl_ph_season_13_teams.csv")
    assert len(teams) == 8 and not teams.team_id.duplicated().any()
    row = teams.loc[teams.team_code.eq(code)].iloc[0]
    assert row.team_name == after and row.historical_names == f"{before} | {after}"


def test_collected_historical_statistics_and_source_hashes_are_auditable() -> None:
    stats = pd.read_csv(ROOT / "data/raw/historical_player_season_stats.csv")
    assert len(stats) > 0 and stats["season"].between(5, 17).all()
    assert set(stats["season"]) == set(range(5, 18))
    assert not stats["feature_eligible_for_training"].any()
    assert stats[["games_played", "player_id", "role"]].isna().all().all()
    canonical = pd.read_parquet(ROOT / "data/processed/canonical/player_season_stats.parquet")
    assert canonical["feature_eligible_for_training"].notna().all()
    assert not canonical["feature_eligible_for_training"].any()
    manifest = pd.read_csv(ROOT / "data/raw/season_evidence_source_manifest.csv")
    for row in manifest.loc[manifest["retrieval_status"].eq("ok")].itertuples():
        assert sha256((ROOT / row.snapshot_path).read_bytes()) == row.sha256
    audit = json.loads(
        (ROOT / "reports/season_reconciliation.json").read_text(),
        parse_constant=lambda value: pytest.fail(f"Nonstandard JSON constant: {value}"),
    )
    assert {row["season"] for row in audit} == set(range(1, 19))
    assert next(row for row in audit if row["season"] == 18)["status"] == (
        "regular_wl_totals_reconciled"
    )


def test_s13_team_catalog_has_one_row_per_identity_and_keeps_historical_labels() -> None:
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    season = matches.loc[matches["season"].eq(13)]
    for side in ("a", "b"):
        selected = season.loc[season[f"team_{side}_code"].eq("TLPH")]
        assert set(selected.loc[selected.stage.eq("regular_season"), f"team_{side}_name"]) <= {
            "ECHO",
        }
        assert set(selected.loc[selected.stage.eq("playoffs"), f"team_{side}_name"]) <= {
            "Team Liquid ECHO PH",
        }
    teams = pd.read_csv(ROOT / "data/mpl-season13/mpl_ph_season_13_teams.csv")
    assert len(teams) == 8 and not teams["team_id"].duplicated().any()
    team = teams.loc[teams.team_code.eq("TLPH")].iloc[0]
    assert team.team_name == "Team Liquid ECHO PH"
    assert team.historical_names == "ECHO | Team Liquid ECHO PH"


def test_cache_only_without_sources_preserves_previous_observations(tmp_path, monkeypatch) -> None:
    from mpl_ph_predictor.data import season_evidence

    for folder in ("data/clean", "data/raw", "config", "reports"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "config/competition.json").write_text(
        json.dumps(
            {
                "data_cutoff": "2026-11-09",
                "timezone": "Asia/Manila",
            }
        )
    )
    (tmp_path / "config/team_name_history.json").write_text('{"transitions": []}')
    (tmp_path / "config/league_history.json").write_text(
        '{"verification_sources": [], "events": []}'
    )
    pd.DataFrame({"season": [8], "stage": ["regular_season"]}).to_csv(
        tmp_path / "data/clean/matches.csv",
        index=False,
    )
    previous = parse_archive_statistics(stats_fixture(), 8, PROVENANCE)
    path = tmp_path / "data/raw/historical_player_season_stats.csv"
    previous.to_csv(path, index=False)
    original = pd.read_csv(path)

    def forbid_network(*args, **kwargs):
        raise AssertionError("Cache-only must not access the network")

    monkeypatch.setattr(season_evidence.requests, "get", forbid_network)
    result = collect_season_evidence(root=tmp_path, cache_only=True)
    assert result["previous_stat_observations_retained"] == 1
    pd.testing.assert_frame_equal(pd.read_csv(path), original)
    assert pd.read_csv(path).iloc[0].player == "fixture_player"
    snapshot_path = tmp_path / "data/raw/fixture_only.html"
    snapshot_path.write_bytes(stats_fixture())
    pd.DataFrame(
        [
            {
                "url": PROVENANCE["url"],
                "retrieval_status": "ok",
                "sha256": "invalid_hash",
                "snapshot_path": str(snapshot_path.relative_to(tmp_path)),
                "season": 8,
            }
        ]
    ).to_csv(tmp_path / "data/raw/season_evidence_source_manifest.csv", index=False)
    result = collect_season_evidence(root=tmp_path, cache_only=True)
    assert result["previous_stat_observations_retained"] == 1
    audit = json.loads((tmp_path / "reports/season_reconciliation.json").read_text())
    assert next(row for row in audit if row["season"] == 8)["status"] == "invalid_cached_snapshot"
    pd.testing.assert_frame_equal(pd.read_csv(path), original)
