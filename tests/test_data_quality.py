import json
from pathlib import Path

import pandas as pd

from mpl_ph_predictor.data import compute_standings

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_counts_and_quality_pass() -> None:
    quality = json.loads((ROOT / "reports" / "data_quality.json").read_text())
    assert quality["status"] == "pass"
    assert quality["canonical_match_rows"] == 852
    assert quality["historical_training_rows_before_s18"] == 788
    assert quality["s8_match_rows"] == 64
    assert quality["corrected_score_rows"] == 3
    assert quality["s18_regular_rows"] == 56
    assert quality["s18_playoff_rows"] == 8
    assert quality["s18_standings_match_official"] is True


def test_s18_standings_recompute_to_official_table() -> None:
    matches = pd.read_csv(ROOT / "data" / "clean" / "matches.csv")
    regular = matches.query("season == 18 and stage == 'regular_season'")
    standings = compute_standings(regular).set_index("team_code")
    assert standings.loc["TLPH", "match_wins"] == 12
    assert standings.loc["FLCN", "net_game_wins"] == 16
    assert standings.loc["TWIS", "game_losses"] == 28
    assert standings.loc["ONIC", "h2h_net_game_wins"] == 1
    assert standings.loc["RORA", "h2h_net_game_wins"] == -1
    assert standings.index.tolist() == [
        "TLPH",
        "FLCN",
        "ONIC",
        "RORA",
        "OMG",
        "TNC",
        "APBR",
        "TWIS",
    ]


def test_source_manifest_records_real_retrieval_dates_and_allowed_domains() -> None:
    sources = pd.read_csv(ROOT / "data" / "raw" / "source_manifest.csv")
    accessed = pd.to_datetime(sources["accessed_at"], utc=True)
    assert accessed.notna().all() and accessed.le(pd.Timestamp.now(tz="UTC")).all()
    cutoff = json.loads((ROOT / "config/competition.json").read_text())["data_cutoff"]
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    dates = pd.to_datetime(matches["scheduled_at"], utc=True).dt.tz_convert("Asia/Manila")
    assert dates.lt(pd.Timestamp(cutoff, tz="Asia/Manila") + pd.Timedelta(days=1)).all()
    from urllib.parse import urlparse

    allowed = {
        "ph-mpl.com",
        "en.moonton.com",
        "teamliquid.com",
        "tiebreakertimes.com.ph",
        "www.dugout.ph",
    }
    assert set(sources["url"].map(lambda url: urlparse(url).hostname)).issubset(allowed)
    assert sources["sha256"].str.len().eq(64).all()
    assert sources["snapshot_path"].map(lambda path: (ROOT / path).is_file()).all()


def test_missing_official_archives_are_documented() -> None:
    issues = json.loads((ROOT / "data" / "raw" / "collection_issues.json").read_text())
    scopes = {issue["scope"] for issue in issues}
    assert {"season_5", "season_6", "roster_features"}.issubset(scopes)


def test_all_eighteen_seasons_have_honest_coverage_and_reference_layout() -> None:
    coverage = pd.read_csv(ROOT / "reports/season_coverage.csv")
    assert coverage["season"].tolist() == list(range(1, 19))
    assert coverage.loc[coverage["season"].le(5), "coverage_status"].eq("partial").all()
    assert coverage.loc[coverage["season"].le(5), "model_eligible_rows"].eq(0).all()
    for season in range(1, 19):
        folder = ROOT / "data" / (f"mpl-season{season}" if season < 18 else "season18")
        assert (folder / f"mpl_ph_season_{season:02}_matches.csv").is_file()
        assert (folder / "coverage.json").is_file()
    canonical = pd.read_parquet(ROOT / "data/processed/canonical/matches.parquet")
    assert len(canonical) == 852


def test_s8_identity_does_not_backdate_team_liquid() -> None:
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    regular = matches.query("season == 8 and stage == 'regular_season'")
    assert len(regular) == 56
    counts = pd.concat([regular["team_a_code"], regular["team_b_code"]]).value_counts()
    assert counts.eq(14).all()
    assert "TLPH" not in counts.index
    assert "ECHO" in counts.index
    assert not regular["team_a_code"].eq(regular["team_b_code"]).any()
    wins = compute_standings(regular).set_index("team_code")["match_wins"].to_dict()
    assert wins == {
        "BLCK": 13,
        "ONIC": 9,
        "NXPE": 7,
        "OMG": 5,
        "ECHO": 6,
        "RSG": 7,
        "BREN": 5,
        "TNC": 4,
    }


def test_roster_unknown_effective_dates_are_not_invented() -> None:
    rosters = pd.read_csv(ROOT / "data/clean/rosters.csv")
    assert rosters["effective_from"].isna().all()
    assert rosters["effective_date_status"].eq("unverified").all()


def test_early_bo2_draws_are_retained_but_not_used_for_binary_training() -> None:
    early = pd.read_csv(ROOT / "data/raw/early_match_evidence.csv")
    assert set(early["season"]) == {1, 2, 3, 4, 5}
    assert early["scheduled_at"].isna().all()
    draws = early.loc[early["team_a_score"].eq(early["team_b_score"])]
    assert len(draws) > 0
    assert not draws["training_eligible"].any()


def test_bo3_incomplete_or_drawn_scores_are_rejected() -> None:
    from mpl_ph_predictor.data.normalization import invalid_completed_score_mask

    fixtures = pd.DataFrame(
        {
            "team_a_score": [0, 1, 2, 2, 2.5, None],
            "team_b_score": [0, 1, 0, 1, 0, 2],
            "best_of": [3] * 6,
        }
    )
    assert invalid_completed_score_mask(fixtures).tolist() == [True, True, False, False, True, True]


def test_s6_corrupt_scores_keep_originals_and_use_verified_outcomes() -> None:
    raw = pd.read_csv(ROOT / "data/raw/official_matches.csv")
    corrected = raw.loc[raw["correction_source_urls"].notna()].set_index("original_score_text")
    assert set(corrected.index) == {"2:-2", "0:0", "1:1"}
    assert corrected.loc["0:0", "team_b_score"] == 2
    assert corrected.loc["1:1", "team_b_score"] == 2
