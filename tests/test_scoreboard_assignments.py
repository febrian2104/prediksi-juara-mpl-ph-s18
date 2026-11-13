"""Reviewed historical HTML fixtures; no invented competition observations."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from mpl_ph_predictor.archive import sha256
from mpl_ph_predictor.data.game_details import parse_game_detail
from mpl_ph_predictor.data.scoreboard_assignments import load_reviewed_assignments

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-11-09"


def reviewed_fixture(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "data/raw/source_snapshots").mkdir(parents=True)
    config = json.loads((ROOT / "config/reviewed_scoreboard_assignments.json").read_text())
    config["series"] = config["series"][:1]
    path = tmp_path / "config/reviewed_scoreboard_assignments.json"
    path.write_text(json.dumps(config))
    entry = config["series"][0]
    manifest = pd.read_csv(ROOT / "data/raw/season_evidence_source_manifest.csv")
    source = entry["sources"][0]
    manifest = manifest.loc[
        manifest.url.eq(source["url"])
        & manifest.sha256.eq(source["reviewed_snapshot_sha256"])
    ].tail(1)
    manifest.to_csv(tmp_path / "data/raw/season_evidence_source_manifest.csv", index=False)
    shutil.copy2(ROOT / manifest.iloc[0].snapshot_path, tmp_path / manifest.iloc[0].snapshot_path)
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    match = matches.loc[matches.match_id.eq(entry["match_id"])].iloc[0].to_dict()
    snapshots = pd.read_csv(ROOT / "data/raw/game_detail_source_manifest.csv")
    record = snapshots.loc[
        snapshots.url.eq(match["source_url"]) & snapshots.sha256.eq(entry["scoreboard_sha256"])
    ].iloc[-1].to_dict()
    return path, match, record, (ROOT / record["snapshot_path"]).read_bytes()


def test_reviewed_grouping_preserves_every_source_row_and_numeric_value():
    reviews = load_reviewed_assignments(ROOT, CUTOFF)
    matches = pd.read_csv(ROOT / "data/clean/matches.csv").set_index("match_id", drop=False)
    manifest = pd.read_csv(ROOT / "data/raw/game_detail_source_manifest.csv")
    games_count, corrected_count = 0, 0
    numeric_cells = {
        "kills": 2, "deaths": 3, "assists": 4, "official_kda": 5,
        "gold": 6, "hero_damage": 10, "damage_taken": 11, "tower_damage": 12,
    }
    assert len(reviews) == 4
    for match_id, review in reviews.items():
        assert review["verified"]
        match = matches.loc[match_id].to_dict()
        record = manifest.loc[
            manifest.url.eq(match["source_url"]) & manifest.sha256.eq(review["scoreboard_sha256"])
        ].iloc[-1].to_dict()
        content = (ROOT / record["snapshot_path"]).read_bytes()
        # Without the explicit review these malformed original tables still fail.
        with pytest.raises(ValueError, match="lima pemain"):
            parse_game_detail(content, match, record)
        games, drafts, players = parse_game_detail(content, match, record, review=review)
        soup = BeautifulSoup(content, "html.parser")
        for row in players:
            pane = soup.select_one(f"#game{row['game_number']}")
            candidates = [
                tr for tr in pane.select("table.table-scoreboard tbody tr")
                if tr.find("td").get_text(strip=True) == row["player"]
            ]
            assert len(candidates) == 1
            tr = candidates[0]
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
            logo = tr.find_parent("div", class_="container").select_one("img[alt$=' logo']")
            assert row["source_team_code"] == logo["alt"][:-5].strip()
            assert row["hero"] == cells[1]
            for column, index in numeric_cells.items():
                assert row[column] == float(cells[index].replace(",", ""))
            assert not row["feature_eligible_for_training"]
            assert not row["feature_eligible_for_s18"]
            if row["team_code"] != row["source_canonical_team_code"]:
                corrected_count += 1
                assert row["assignment_registry_sha256"] == review["registry_sha256"]
                assert row["assignment_reviewed_on"] == "2026-11-13"
        assert len(players) == len(drafts) == len(games) * 10
        assert all(g["winner"] is None and g["duration_minutes"] is None for g in games)
        assert all(d["order"] is None for d in drafts)
        assert pd.DataFrame(players).groupby(["game_number", "team_code"]).size().eq(5).all()
        games_count += len(games)
    assert games_count == 10 and corrected_count == 12


@pytest.mark.parametrize("change", ["scoreboard", "date", "opponent", "score", "match_id"])
def test_review_cannot_be_reused_for_different_source_or_match(tmp_path, change):
    _, match, record, content = reviewed_fixture(tmp_path)
    review = load_reviewed_assignments(tmp_path, CUTOFF)[match["match_id"]]
    if change == "scoreboard":
        content += b"<!-- unreviewed revision -->"
    elif change == "date":
        match["scheduled_at"] = "2021-08-28T08:00:00Z"
    elif change == "opponent":
        match["team_b_code"] = "OMG"
    elif change == "score":
        match["team_b_score"] = 1
    else:
        match["match_id"] = "not_the_reviewed_series"
    with pytest.raises(ValueError):
        parse_game_detail(content, match, record, review=review)


@pytest.mark.parametrize("change", ["missing", "corrupt", "wrong_content", "failed_download"])
def test_missing_or_invalid_corroboration_disables_correction(tmp_path, change):
    config_path, match, record, content = reviewed_fixture(tmp_path)
    path = tmp_path / "data/raw/season_evidence_source_manifest.csv"
    manifest = pd.read_csv(path)
    snapshot = tmp_path / manifest.iloc[0].snapshot_path
    if change == "missing":
        manifest.loc[0, "snapshot_path"] = "data/raw/source_snapshots/absent-test-fixture.html"
    elif change == "failed_download":
        manifest.loc[0, "retrieval_status"] = "failed"
    else:
        replacement = b"<html>Unrelated test fixture, not competition evidence.</html>"
        snapshot.write_bytes(replacement)
        if change == "wrong_content":
            manifest.loc[0, "sha256"] = sha256(replacement)
            config = json.loads(config_path.read_text())
            config["series"][0]["sources"][0]["reviewed_snapshot_sha256"] = sha256(replacement)
            config_path.write_text(json.dumps(config))
    manifest.to_csv(path, index=False)
    review = load_reviewed_assignments(tmp_path, CUTOFF)[match["match_id"]]
    assert not review["verified"]
    with pytest.raises(ValueError, match="Bukti review"):
        parse_game_detail(content, match, record, review=review)


@pytest.mark.parametrize("change", ["player", "source_team", "target_team", "missing_row"])
def test_review_never_invents_rows_or_accepts_incomplete_grouping(tmp_path, change):
    _, match, record, content = reviewed_fixture(tmp_path)
    review = load_reviewed_assignments(tmp_path, CUTOFF)[match["match_id"]]
    assignment = review["assignments"][0]
    if change == "player":
        assignment["player"] = "unknown_test_fixture_player"
    elif change == "source_team":
        assignment["from_raw_team_code"] = "ONIC"
    elif change == "target_team":
        assignment["to_team_code"] = "ECHO"
    else:
        soup = BeautifulSoup(content, "html.parser")
        soup.select_one("#game1 table.table-scoreboard tbody tr").decompose()
        content = str(soup).encode()
        # Even an explicitly re-reviewed checksum cannot supply a missing row.
        review["scoreboard_sha256"] = sha256(content)
    with pytest.raises(ValueError):
        parse_game_detail(content, match, record, review=review)


@pytest.mark.parametrize("change", ["training", "cutoff", "duplicate", "game", "future_source"])
def test_invalid_assignment_registry_is_rejected_before_publication(tmp_path, change):
    path, _, _, _ = reviewed_fixture(tmp_path)
    config = json.loads(path.read_text())
    if change == "training":
        config["feature_eligible"] = True
    elif change == "cutoff":
        config["information_cutoff"] = "2026-11-13"
    elif change == "duplicate":
        config["series"].append(config["series"][0])
    elif change == "game":
        config["series"][0]["assignments"][0]["game_numbers"] = [True]
    else:
        config["series"][0]["sources"][0]["published_on"] = "2026-11-14"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_reviewed_assignments(tmp_path, CUTOFF)


def test_newer_article_does_not_replace_pinned_reviewed_evidence(tmp_path):
    _, match, _, _ = reviewed_fixture(tmp_path)
    path = tmp_path / "data/raw/season_evidence_source_manifest.csv"
    manifest = pd.read_csv(path)
    newer = manifest.iloc[0].to_dict()
    replacement = b"<html>New unreviewed revision of the test article.</html>"
    newer.update(
        sha256=sha256(replacement),
        snapshot_path="data/raw/source_snapshots/new-test-article.html",
        retrieved_at="2026-11-13T03:00:00Z",
    )
    (tmp_path / newer["snapshot_path"]).write_bytes(replacement)
    pd.concat([manifest, pd.DataFrame([newer])], ignore_index=True).to_csv(path, index=False)
    review = load_reviewed_assignments(tmp_path, CUTOFF)[match["match_id"]]
    assert review["verified"]
    assert review["evidence"][0]["sha256"] == manifest.iloc[0].sha256


def test_counter_audit_uses_the_observation_snapshot_not_the_latest_download(tmp_path):
    from mpl_ph_predictor.data.game_audit import audit_game_summaries

    _, match, record, content = reviewed_fixture(tmp_path)
    (tmp_path / "data/clean").mkdir()
    pd.DataFrame([match]).to_csv(tmp_path / "data/clean/matches.csv", index=False)
    review = load_reviewed_assignments(tmp_path, CUTOFF)[match["match_id"]]
    games, _, players = parse_game_detail(content, match, record, review=review)
    pd.DataFrame(games).to_csv(tmp_path / "data/raw/official_games.csv", index=False)
    pd.DataFrame(players).to_csv(tmp_path / "data/raw/official_player_game_stats.csv", index=False)
    (tmp_path / record["snapshot_path"]).write_bytes(content)
    replacement = b"<html>Later test revision with no game scoreboard.</html>"
    newer = {
        **record,
        "sha256": sha256(replacement),
        "snapshot_path": "data/raw/source_snapshots/new-test-scoreboard.html",
    }
    (tmp_path / newer["snapshot_path"]).write_bytes(replacement)
    pd.DataFrame([record, newer]).to_csv(
        tmp_path / "data/raw/game_detail_source_manifest.csv", index=False
    )
    report = audit_game_summaries(root=tmp_path)
    assert report["comparable_counters"] == 20
    assert report["counter_differences"] == report["games_without_complete_comparison"] == 0
    checks = pd.read_csv(tmp_path / "reports/game_summary_checks.csv")
    assert set(checks.source_snapshot_path) == {record["snapshot_path"]}
