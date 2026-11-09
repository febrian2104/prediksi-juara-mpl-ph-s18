import json
from pathlib import Path

import pandas as pd

from mpl_ph_predictor.data import compute_standings

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_counts_and_quality_pass() -> None:
    quality = json.loads((ROOT / "reports" / "data_quality.json").read_text())
    assert quality["status"] == "pass"
    assert quality["canonical_match_rows"] == 787
    assert quality["historical_training_rows_before_s18"] == 723
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


def test_source_manifest_uses_cutoff_and_official_domains() -> None:
    sources = pd.read_csv(ROOT / "data" / "raw" / "source_manifest.csv")
    assert set(sources["accessed_at"]) == {"2026-11-09"}
    assert sources["url"].str.startswith(("https://ph-mpl.com", "https://en.moonton.com")).all()


def test_missing_official_archives_are_documented() -> None:
    issues = json.loads((ROOT / "data" / "raw" / "collection_issues.json").read_text())
    scopes = {issue["scope"] for issue in issues}
    assert {"season_5", "season_8", "season_6", "roster_features"}.issubset(scopes)
