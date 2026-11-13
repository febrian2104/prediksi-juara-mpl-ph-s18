"""Offline checks of reviewed metadata, never invented player identities."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mpl_ph_predictor.archive import sha256
from mpl_ph_predictor.data.catalog import official_player_stats, player_table
from mpl_ph_predictor.data.player_aliases import METADATA_COLUMNS, STATUS, annotate_player_aliases

ROOT = Path(__file__).resolve().parents[1]


def source_frames():
    stats = pd.read_csv(ROOT / "data/raw/official_s18_player_season_stats.csv")
    manifest = pd.read_csv(ROOT / "data/raw/source_manifest.csv")
    record = manifest.loc[manifest.url.eq("https://ph-mpl.com/data")].iloc[0]
    stats["source_snapshot_sha256"] = record.sha256
    # Reconstruct the pre-review unresolved state even after production exports update.
    mask = stats.player.isin(["JMPINKMAN", "Domeng", "Sensu1", "Vinnn"])
    stats.loc[mask, "player_identity_status"] = "unresolved_ign_alias"
    stats.loc[mask, ["player_id", "matched_roster_player", "role"]] = pd.NA
    for column in METADATA_COLUMNS:
        if column in stats:
            stats[column] = pd.NA
    players = player_table(pd.read_csv(ROOT / "data/clean/rosters.csv"))
    return stats, players


def fixture(tmp_path, code="APBR"):
    (tmp_path / "config").mkdir()
    (tmp_path / "data/raw/source_snapshots").mkdir(parents=True)
    config = json.loads((ROOT / "config/reviewed_player_aliases.json").read_text())
    config["aliases"] = [entry for entry in config["aliases"] if entry["team_code"] == code]
    path = tmp_path / "config/reviewed_player_aliases.json"
    path.write_text(json.dumps(config))
    shutil.copy2(ROOT / "config/competition.json", tmp_path / "config/competition.json")
    entry = config["aliases"][0]
    for name, sources in (
        ("source_manifest", [config["statistics_source"], entry["roster_source"]]),
        ("season_evidence_source_manifest", entry["supporting_sources"]),
    ):
        manifest = pd.read_csv(ROOT / f"data/raw/{name}.csv")
        selected = [
            manifest.loc[
                manifest.url.eq(source["url"])
                & manifest.sha256.eq(source["reviewed_snapshot_sha256"])
            ].tail(1)
            for source in sources
        ]
        if not selected:
            continue
        combined = pd.concat(selected, ignore_index=True)
        combined.to_csv(tmp_path / f"data/raw/{name}.csv", index=False)
        for relative in combined.snapshot_path:
            shutil.copy2(ROOT / relative, tmp_path / relative)
    stats, players = source_frames()
    return stats, players, path


def test_reviewed_aliases_preserve_source_igns_numbers_and_temporal_limits():
    stats, players = source_frames()
    original = stats.copy(deep=True)
    result, audit = annotate_player_aliases(stats, players)
    assert audit["status"] == "pass" and audit["resolved_aliases"] == 4
    assert audit["unresolved_ign_rows"] == 0 and not audit["feature_eligible"]
    reviewed = result.loc[result.player_identity_status.eq(STATUS)].set_index("player")
    assert reviewed.matched_roster_player.to_dict() == {
        "JMPINKMAN": "JIMPINKMAN", "Domeng": "Domengkite", "Sensu1": "Sensui", "Vinnn": "Vin",
    }
    assert reviewed.player_alias_basis.value_counts().to_dict() == {
        "reviewed_official_roster_asset_alias": 3,
        "reviewed_contemporary_reports_same_full_name": 1,
    }
    mutable = {"role", "player_id", "matched_roster_player", "player_identity_status"}
    immutable = [c for c in stats if c not in mutable and c not in METADATA_COLUMNS]
    pd.testing.assert_frame_equal(result[immutable], original[immutable])
    pd.testing.assert_frame_equal(stats, original)
    assert not result.feature_eligible_for_training.any()
    assert not result.feature_eligible_for_s18.any()
    assert players.valid_from.isna().all() and players.valid_to.isna().all()
    assert set(reviewed.player_alias_reviewed_on) == {"2026-11-13"}


@pytest.mark.parametrize("change", ["season", "team", "duplicate", "missing", "coach"])
def test_alias_link_is_not_global_or_reused_for_ambiguous_roster(tmp_path, change):
    stats, players, _ = fixture(tmp_path)
    mask = players.team_code.eq("APBR") & players.player.eq("JIMPINKMAN")
    if change == "season":
        players.loc[mask, "season"] = 17
    elif change == "team":
        players.loc[mask, "team_code"] = "TNC"
    elif change == "duplicate":
        players = pd.concat([players, players.loc[mask]], ignore_index=True)
    elif change == "missing":
        players = players.loc[~mask]
    else:
        players.loc[mask, "member_type"] = "coach"
    result, audit = annotate_player_aliases(stats, players, root=tmp_path)
    assert audit["resolved_aliases"] == 0
    row = result.loc[result.player.eq("JMPINKMAN")].iloc[0]
    assert pd.isna(row.player_id) and pd.isna(row.role)


@pytest.mark.parametrize("change", ["new_stats", "duplicate_stats", "collision", "eligible"])
def test_source_scope_and_existing_observations_are_not_overwritten(tmp_path, change):
    stats, players, _ = fixture(tmp_path)
    mask = stats.player.eq("JMPINKMAN")
    if change == "new_stats":
        stats.loc[mask, "source_snapshot_sha256"] = "0" * 64
    elif change == "duplicate_stats":
        stats = pd.concat([stats, stats.loc[mask]], ignore_index=True)
    elif change == "collision":
        duplicate = stats.loc[mask].copy()
        duplicate["player"] = "JIMPINKMAN"
        duplicate["player_id"] = "jimpinkman"
        stats = pd.concat([stats, duplicate], ignore_index=True)
    else:
        stats.loc[mask, "feature_eligible_for_training"] = True
    result, audit = annotate_player_aliases(stats, players, root=tmp_path)
    assert len(result) == len(stats) and audit["resolved_aliases"] == 0
    assert result.loc[result.player.eq("JMPINKMAN"), "player_id"].isna().all()


@pytest.mark.parametrize("change", ["corrupt", "missing", "asset", "content", "published"])
def test_failed_evidence_clears_previously_reviewed_links(tmp_path, change):
    code = "TNC" if change in ("content", "published") else "APBR"
    stats, players, path = fixture(tmp_path, code)
    prior, before = annotate_player_aliases(stats, players, root=tmp_path)
    assert before["resolved_aliases"] == 1
    config = json.loads(path.read_text())
    entry = config["aliases"][0]
    if change in ("corrupt", "missing"):
        manifest_path = tmp_path / "data/raw/source_manifest.csv"
        manifest = pd.read_csv(manifest_path)
        index = manifest.index[manifest.url.eq(entry["roster_source"]["url"])][0]
        snapshot = tmp_path / manifest.loc[index, "snapshot_path"]
        if change == "corrupt":
            snapshot.write_bytes(b"<html>Corrupt test fixture.</html>")
        else:
            manifest.loc[index, "snapshot_path"] = "data/raw/source_snapshots/missing-test.html"
            manifest.to_csv(manifest_path, index=False)
    elif change == "asset":
        entry["roster_source"]["asset_path"] = "/season18/player/APBR/unrelated-test.png"
    elif change == "content":
        entry["supporting_sources"][0]["required_terms"] = ["Absent test-only phrase"]
    else:
        entry["supporting_sources"][0]["published_on"] = "2026-08-18"
    path.write_text(json.dumps(config))
    result, audit = annotate_player_aliases(prior, players, root=tmp_path)
    assert audit["resolved_aliases"] == 0
    row = result.loc[result.player.eq(entry["statistics_ign"])].iloc[0]
    assert row.player_identity_status == "unresolved_ign_alias"
    assert pd.isna(row.player_id) and pd.isna(row.player_alias_registry_sha256)


@pytest.mark.parametrize("change", ["training", "cutoff", "duplicate", "target", "future"])
def test_invalid_registry_is_rejected(tmp_path, change):
    stats, players, path = fixture(tmp_path, "TNC")
    config = json.loads(path.read_text())
    if change == "training":
        config["feature_eligible"] = True
    elif change == "cutoff":
        config["information_cutoff"] = "2026-11-13"
    elif change == "duplicate":
        config["aliases"].append(config["aliases"][0])
    elif change == "target":
        duplicate = {**config["aliases"][0], "statistics_ign": "TestAlias"}
        config["aliases"].append(duplicate)
    else:
        config["aliases"][0]["supporting_sources"][0]["published_on"] = "2026-11-14"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        annotate_player_aliases(stats, players, root=tmp_path)


def test_absent_registry_does_not_guess_aliases(tmp_path):
    stats, players = source_frames()
    result, audit = annotate_player_aliases(stats, players, root=tmp_path)
    assert audit["status"] == "registry_not_available"
    assert result.player_identity_status.eq("unresolved_ign_alias").sum() == 4


def test_new_article_revision_cannot_replace_reviewed_snapshot(tmp_path):
    stats, players, _ = fixture(tmp_path, "TNC")
    path = tmp_path / "data/raw/season_evidence_source_manifest.csv"
    manifest = pd.read_csv(path)
    newer = manifest.iloc[0].to_dict()
    content = b"<html>New unreviewed test article.</html>"
    newer.update(sha256=sha256(content), snapshot_path="data/raw/source_snapshots/new-test.html")
    (tmp_path / newer["snapshot_path"]).write_bytes(content)
    pd.concat([manifest, pd.DataFrame([newer])], ignore_index=True).to_csv(path, index=False)
    _, audit = annotate_player_aliases(stats, players, root=tmp_path)
    assert audit["resolved_aliases"] == 1
    assert audit["aliases"][0]["sources"][2]["sha256"] == manifest.iloc[0].sha256


def test_statistics_parser_validates_checksum_and_retains_raw_igns(tmp_path):
    _, players, _ = fixture(tmp_path)
    result = official_player_stats(players, root=tmp_path)
    assert len(result) == 49 and result.player_identity_status.eq(STATUS).sum() == 1
    manifest = pd.read_csv(tmp_path / "data/raw/source_manifest.csv")
    source = manifest.loc[manifest.url.eq("https://ph-mpl.com/data")].iloc[0]
    snapshot = tmp_path / source.snapshot_path
    snapshot.write_bytes(b"<html>Unreviewed/corrupt test statistics.</html>")
    with pytest.raises(ValueError, match="Checksum"):
        official_player_stats(players, root=tmp_path)


@pytest.mark.parametrize(
    "change", ["missing", "table", "headers", "nan", "inf", "negative", "percent"]
)
def test_bad_statistics_never_publish_empty_or_mislabelled_observations(tmp_path, change):
    _, players, _ = fixture(tmp_path)
    manifest_path = tmp_path / "data/raw/source_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    index = manifest.index[manifest.url.eq("https://ph-mpl.com/data")][0]
    snapshot = tmp_path / manifest.loc[index, "snapshot_path"]
    if change == "missing":
        manifest.loc[index, "snapshot_path"] = "data/raw/source_snapshots/missing-test.html"
    else:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(snapshot.read_bytes(), "html.parser")
        table = next(t for t in soup.select("table") if "Kill Participation" in t.get_text())
        if change == "table":
            table.decompose()
        elif change == "headers":
            table.select("thead th")[1].string = "Unknown Team Column"
        else:
            values = {"nan": "NaN", "inf": "inf", "negative": "-1", "percent": "101%"}
            table.select_one("tbody tr").find_all("td")[9].string = values[change]
        content = str(soup).encode()
        snapshot.write_bytes(content)
        manifest.loc[index, "sha256"] = sha256(content)
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError):
        official_player_stats(players, root=tmp_path)
