"""Offline parser fixtures are synthetic tests, never exported as competition data."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mpl_ph_predictor.archive import sha256
from mpl_ph_predictor.data.game_details import collect_game_details, parse_game_detail

ROOT = Path(__file__).resolve().parents[1]
HEADERS = [
    "Player",
    "Hero",
    "Kill",
    "Death",
    "Assist",
    "KDA",
    "Gold",
    "Items",
    "Emblems",
    "Talents",
    "Hero Damage",
    "Damage Taken",
    "Tower Damage",
]


def fixture_html() -> bytes:
    tables = []
    for code in ("FIXTURE_A", "FIXTURE_B"):
        headers = "".join(f"<th>{header}</th>" for header in HEADERS)
        rows = []
        for index in range(5):
            cells = [
                f"test_player_{index}",
                f"test_hero_{index}",
                "0",
                "0",
                "0",
                "0",
                "1,000",
                "",
                "",
                "",
                "100",
                "200",
                "0",
            ]
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
        tables.append(
            f'<div class="container"><img alt="{code} logo">'
            f'<table class="table-scoreboard"><thead><tr>{headers}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    return ('<div id="game1">' + "".join(tables) + "</div>").encode()


def fixture_match() -> dict:
    return {
        "season": 18,
        "match_id": "fixture_only",
        "scheduled_at": "2026-08-21T08:00:00Z",
        "team_a_code": "FIXTURE_A",
        "team_b_code": "FIXTURE_B",
        "team_a_name": "Test A",
        "team_b_name": "Test B",
        "team_a_score": 1,
        "team_b_score": 0,
        "source_url": "https://ph-mpl.com/data/match/fixture-only",
    }


def fixture_provenance() -> dict:
    return {
        "snapshot_path": "fixture_only.html",
        "retrieved_at": "2026-11-10T01:00:00Z",
        "information_cutoff": "2026-11-09",
    }


def test_observed_picks_do_not_invent_winners_durations_or_draft_order() -> None:
    games, drafts, players = parse_game_detail(
        fixture_html(),
        fixture_match(),
        fixture_provenance(),
    )
    assert len(games) == 1
    assert len(drafts) == len(players) == 10
    assert games[0]["winner"] is None
    assert games[0]["duration_minutes"] is None
    assert all(row["order"] is None for row in drafts)
    assert {row["action"] for row in drafts} == {"pick"}
    assert all(not row["feature_eligible_for_s18"] for row in players)
    assert players[0]["gold"] == 1000
    assert players[0]["observed_at"].startswith("2026-11-10")
    assert players[0]["information_cutoff"] == "2026-11-09"


@pytest.mark.parametrize("mutation", ["game_count", "team", "headers", "players"])
def test_invalid_detail_is_rejected_atomically(mutation: str) -> None:
    content = fixture_html()
    match = fixture_match()
    if mutation == "game_count":
        match["team_a_score"] = 2
    elif mutation == "team":
        content = content.replace(b"FIXTURE_B logo", b"OTHER_TEAM logo")
    elif mutation == "headers":
        content = content.replace(b"Tower Damage", b"Unknown Column")
    elif mutation == "players":
        content = content.replace(b"<td>test_player_4</td>", b"<td>test_player_0</td>")
    with pytest.raises(ValueError):
        parse_game_detail(content, match, fixture_provenance())


def test_cache_only_rejects_bad_hash_and_never_accesses_network(tmp_path, monkeypatch) -> None:
    from mpl_ph_predictor.data import game_details

    for folder in ("data/clean", "data/raw/source_snapshots", "config", "reports"):
        (tmp_path / folder).mkdir(parents=True)
    pd.DataFrame([fixture_match()]).to_csv(tmp_path / "data/clean/matches.csv", index=False)
    (tmp_path / "config/competition.json").write_text(
        json.dumps(
            {
                "data_cutoff": "2026-11-09",
                "timezone": "Asia/Manila",
            }
        )
    )
    path = tmp_path / "data/raw/source_snapshots/test.html"
    path.write_bytes(fixture_html())
    record = {
        "url": fixture_match()["source_url"],
        "retrieval_status": "ok",
        "sha256": "bad_hash",
        "snapshot_path": str(path.relative_to(tmp_path)),
        **fixture_provenance(),
    }
    record["snapshot_path"] = str(path.relative_to(tmp_path))
    manifest_path = tmp_path / "data/raw/game_detail_source_manifest.csv"
    pd.DataFrame([record]).to_csv(manifest_path, index=False)

    def forbid_network(*args, **kwargs):
        raise AssertionError("Cache-only must not make network requests")

    monkeypatch.setattr(game_details.requests, "get", forbid_network)
    report = collect_game_details(root=tmp_path, cache_only=True)
    assert report["series_unavailable_or_failed"] == 1
    assert report["games"] == 0
    record["sha256"] = sha256(fixture_html())
    pd.DataFrame([record]).to_csv(manifest_path, index=False)
    report = collect_game_details(root=tmp_path, cache_only=True)
    assert report["series_passed"] == report["games"] == 1
    saved_paths = [
        tmp_path / f"data/raw/official_{name}.csv"
        for name in (
            "games",
            "drafts",
            "player_game_stats",
        )
    ]
    saved_frames = [pd.read_csv(path) for path in saved_paths]
    record["sha256"] = "invalid_after_success"
    pd.DataFrame([record]).to_csv(manifest_path, index=False)
    report = collect_game_details(root=tmp_path, cache_only=True)
    assert report["games"] == 1 and report["retained_previous_games"] == 1
    assert report["series_unavailable_or_failed"] == 1
    for path, original in zip(saved_paths, saved_frames, strict=True):
        pd.testing.assert_frame_equal(pd.read_csv(path), original)


@pytest.mark.parametrize("value", ["NaN", "inf", "-1"])
def test_nonfinite_and_negative_game_statistics_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="negatif/nonfinite"):
        parse_game_detail(
            fixture_html().replace(b"1,000", value.encode()),
            fixture_match(),
            fixture_provenance(),
        )


def test_collected_scoreboards_have_matching_series_counts_and_provenance() -> None:
    games = pd.read_csv(ROOT / "data/raw/official_games.csv")
    drafts = pd.read_csv(ROOT / "data/raw/official_drafts.csv")
    players = pd.read_csv(ROOT / "data/raw/official_player_game_stats.csv")
    matches = pd.read_csv(ROOT / "data/clean/matches.csv").set_index("match_id")
    assert len(games) > 0
    assert len(drafts) == len(players) == len(games) * 10
    assert not games.duplicated(["match_id", "game_number"]).any()
    assert games["winner"].isna().all() and games["duration_minutes"].isna().all()
    assert drafts["order"].isna().all()
    assert set(drafts["action"]) == {"pick"}
    assert not players["feature_eligible_for_s18"].any()
    for match_id, group in games.groupby("match_id"):
        match = matches.loc[match_id]
        assert len(group) == match.team_a_score + match.team_b_score
    manifest = pd.read_csv(ROOT / "data/raw/game_detail_source_manifest.csv")
    for row in manifest.loc[manifest["retrieval_status"].eq("ok")].itertuples():
        assert sha256((ROOT / row.snapshot_path).read_bytes()) == row.sha256
