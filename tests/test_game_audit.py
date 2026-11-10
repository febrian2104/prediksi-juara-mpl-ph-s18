import json
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from mpl_ph_predictor.data.game_audit import METRICS, compare_game_counters

ROOT = Path(__file__).resolve().parents[1]


def source_fixture(gold: str = "5,000"):
    rows = []
    for label, metric in METRICS.items():
        value = gold if metric == "gold" else "500" if metric == "hero_damage" else "0"
        rows.append(
            f'<div><div class="stats-num">{value}</div><div class="stats-divider">{label}</div>'
            f'<div class="stats-num">{value}</div></div>'
        )
    html = (
        '<div id="game1"><div id="summary">'
        + (
            '<div class="team-name"><h3>FIXTURE_A</h3></div>'
            '<div class="team-name"><h3>FIXTURE_B</h3></div>' + "".join(rows)
        )
        + "</div></div>"
    )
    return BeautifulSoup(html, "html.parser").select_one("#game1")


def players_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_code": ["FIXTURE_A"] * 5 + ["FIXTURE_B"] * 5,
            "kills": [0] * 10,
            "deaths": [0] * 10,
            "assists": [0] * 10,
            "gold": [1000] * 10,
            "hero_damage": [100] * 10,
        }
    )


def test_counter_differences_preserve_both_values_without_correction() -> None:
    players = players_fixture()
    original = players.copy(deep=True)
    aliases = {"FIXTURE_A": "FIXTURE_A", "FIXTURE_B": "FIXTURE_B"}
    checks = compare_game_counters(source_fixture(), players, aliases)
    assert len(checks) == 10 and all(c["status"] == "match" for c in checks)
    checks = compare_game_counters(source_fixture("4,999"), players, aliases)
    differences = [c for c in checks if c["status"] == "difference_requires_review"]
    assert len(differences) == 2
    assert all(c["summary_value"] == 4999 and c["scoreboard_sum"] == 5000 for c in differences)
    pd.testing.assert_frame_equal(players, original)


def test_unknown_counter_is_not_treated_as_zero_or_a_match() -> None:
    players = players_fixture()
    aliases = {"FIXTURE_A": "FIXTURE_A", "FIXTURE_B": "FIXTURE_B"}
    checks = compare_game_counters(source_fixture("-"), players, aliases)
    gold = [c for c in checks if c["metric"] == "gold"]
    assert all(c["status"] == "not_comparable" and c["summary_value"] is None for c in gold)
    players.loc[0, "kills"] = pd.NA
    checks = compare_game_counters(source_fixture(), players, aliases)
    unknown = next(c for c in checks if c["team_code"] == "FIXTURE_A" and c["metric"] == "kills")
    assert unknown["scoreboard_sum"] is None and unknown["status"] == "not_comparable"


def test_real_counter_audit_is_complete_and_quality_flags_reach_canonical_games() -> None:
    games = pd.read_csv(ROOT / "data/raw/official_games.csv")
    audit = pd.read_csv(ROOT / "reports/game_summary_audit.csv")
    checks = pd.read_csv(ROOT / "reports/game_summary_checks.csv")
    summary = json.loads((ROOT / "reports/game_summary_audit.json").read_text())
    assert len(audit) == len(games) and len(checks) == len(games) * 10
    assert summary["games_without_complete_comparison"] == 0
    assert summary["counter_differences"] > 0
    assert set(checks.loc[checks.metric.isin(["kills", "deaths", "assists"]), "status"]) == {
        "match"
    }
    assert set(checks.loc[checks.status.eq("difference_requires_review"), "metric"]) <= {
        "gold",
        "hero_damage",
    }
    canonical = pd.read_parquet(ROOT / "data/processed/canonical/games.parquet")
    assert canonical.summary_check_status.notna().all()
    assert canonical.summary_difference_count.sum() == summary["counter_differences"]
