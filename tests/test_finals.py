"""Verification fixtures use copied evidence, never fabricated competition results."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mpl_ph_predictor.archive import sha256
from mpl_ph_predictor.data.catalog import championship_table, match_table, team_table
from mpl_ph_predictor.data.finals import BASIS, annotate_reviewed_finals

ROOT = Path(__file__).resolve().parents[1]


def test_all_reviewed_finals_agree_with_official_cards_without_mutating_matches():
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    before = matches.copy(deep=True)
    annotated, report = annotate_reviewed_finals(matches)
    assert report["status"] == "pass"
    assert report["verified_final_rows"] == 12
    assert {row["season"] for row in report["finals"]} == set(range(6, 18))
    assert not report["feature_eligible"]
    assert annotated["round"].eq("grand_final").sum() == 12
    for row in report["finals"]:
        assert row["official_card"]["status"] == "official_card_verified"
        assert all(source["status"] == "checksum_verified" for source in row["article_sources"])
    pd.testing.assert_frame_equal(matches, before)
    pd.testing.assert_frame_equal(annotated[before.columns], before)
    for season in range(6, 18):
        subset = match_table(annotated.loc[annotated.season.eq(season)])
        # Deliberately move a non-final playoff row to the end.
        shuffled = pd.concat(
            [
                subset.loc[subset["round"].eq("grand_final")],
                subset.loc[~subset["round"].eq("grand_final")],
            ]
        )
        result = championship_table(shuffled, team_table(subset))
        assert result.champion.sum() == result.runner_up.sum() == 1
        assert result.grand_final_basis.eq(BASIS).all()
        assert result.grand_final_match_id.nunique() == 1


def test_raw_aliases_and_article_verification_limits_remain_explicit():
    rows = pd.read_csv(ROOT / "data/clean/matches.csv")
    _, report = annotate_reviewed_finals(rows)
    by_season = {row["season"]: row for row in report["finals"]}
    assert not by_season[6]["article_score_verified"]
    assert not by_season[17]["article_score_verified"]
    assert by_season[12]["article_sources"][0]["kind"] == "secondary_contemporary_report"
    assert by_season[14]["article_sources"][0]["kind"] == "secondary_contemporary_report"
    for season in (10, 11):
        assert by_season[season]["official_card"]["team_code_comparison"] == (
            "official_raw_codes_before_season_scoped_aliases"
        )
        final = rows.loc[rows.match_id.eq(by_season[season]["match_id"])].iloc[0]
        assert final.team_b_code == "ECHO" and final.raw_team_b_code == "TLPH"


def _s13_fixture(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "data/raw/source_snapshots").mkdir(parents=True)
    config = json.loads((ROOT / "config/reviewed_grand_finals.json").read_text())
    config["finals"] = [row for row in config["finals"] if row["season"] == 13]
    path = tmp_path / "config/reviewed_grand_finals.json"
    path.write_text(json.dumps(config))
    for name, url in (
        ("source_manifest", "https://ph-mpl.com/s13"),
        ("season_evidence_source_manifest", "https://en.moonton.com/news/93.html"),
    ):
        manifest = pd.read_csv(ROOT / f"data/raw/{name}.csv")
        manifest = manifest.loc[manifest.url.eq(url)].tail(1)
        manifest.to_csv(tmp_path / f"data/raw/{name}.csv", index=False)
        for relative in manifest.snapshot_path:
            shutil.copy2(ROOT / relative, tmp_path / relative)
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    return matches.loc[matches.season.eq(13)].copy(), path


@pytest.mark.parametrize(
    "change", ["score", "opponent", "date", "time", "official_id", "duplicate"]
)
def test_disagreeing_match_cannot_receive_final_marker(tmp_path, change):
    rows, path = _s13_fixture(tmp_path)
    entry = json.loads(path.read_text())["finals"][0]
    mask = rows.match_id.eq(entry["match_id"])
    if change == "score":
        rows.loc[mask, "team_b_score"] = 1
    elif change == "opponent":
        rows.loc[mask, "team_b_code"] = "BLCK"
    elif change == "date":
        rows.loc[mask, "scheduled_at"] = "2024-05-27T10:00:00Z"
    elif change == "time":
        rows.loc[mask, "scheduled_at"] = "2024-05-26T11:00:00Z"
    elif change == "official_id":
        rows.loc[mask, "official_match_id"] = "unknown_test_fixture_id"
    else:
        rows = pd.concat([rows, rows.loc[mask]], ignore_index=True)
    result, report = annotate_reviewed_finals(rows, root=tmp_path)
    assert report["status"] == "requires_review"
    assert not result["round"].eq("grand_final").any()
    assert result.grand_final_basis.isna().all()


@pytest.mark.parametrize("change", ["corrupt", "missing", "wrong_content", "failed_download"])
def test_source_failures_do_not_preserve_previously_verified_marker(tmp_path, change):
    rows, config_path = _s13_fixture(tmp_path)
    prior, before = annotate_reviewed_finals(rows, root=tmp_path)
    assert before["status"] == "pass"
    path = tmp_path / "data/raw/season_evidence_source_manifest.csv"
    manifest = pd.read_csv(path)
    snapshot = tmp_path / manifest.iloc[0].snapshot_path
    if change == "missing":
        # Point to a missing fixture rather than delete real archived evidence.
        manifest.loc[0, "snapshot_path"] = "data/raw/source_snapshots/missing-fixture.html"
    elif change == "failed_download":
        manifest.loc[0, "retrieval_status"] = "failed"
    else:
        content = b"<html><body>Unrelated test fixture, not match evidence.</body></html>"
        snapshot.write_bytes(content)
        if change == "wrong_content":
            manifest.loc[0, "sha256"] = sha256(content)
            config = json.loads(config_path.read_text())
            config["finals"][0]["sources"][0]["reviewed_snapshot_sha256"] = sha256(content)
            config_path.write_text(json.dumps(config))
    manifest.to_csv(path, index=False)
    result, report = annotate_reviewed_finals(prior, root=tmp_path)
    assert report["status"] == "requires_review"
    assert not result["round"].eq("grand_final").any()
    assert result.grand_final_basis.isna().all()


def test_later_unreviewed_download_cannot_replace_the_reviewed_snapshot(tmp_path):
    rows, config_path = _s13_fixture(tmp_path)
    reviewed = json.loads(config_path.read_text())["finals"][0]["sources"][0]
    path = tmp_path / "data/raw/season_evidence_source_manifest.csv"
    manifest = pd.read_csv(path)
    newer = manifest.iloc[0].to_dict()
    content = b"<html>New unrelated fixture; not the human-reviewed final report.</html>"
    newer["sha256"] = sha256(content)
    newer["snapshot_path"] = f"data/raw/source_snapshots/{newer['sha256']}.html"
    newer["retrieved_at"] = "2026-11-10T06:50:00Z"
    (tmp_path / newer["snapshot_path"]).write_bytes(content)
    pd.concat([manifest, pd.DataFrame([newer])], ignore_index=True).to_csv(path, index=False)
    _, report = annotate_reviewed_finals(rows, root=tmp_path)
    assert report["status"] == "pass"
    used = report["finals"][0]["article_sources"][0]
    assert used["sha256"] == reviewed["reviewed_snapshot_sha256"]
    assert used["sha256"] != newer["sha256"]


@pytest.mark.parametrize("change", ["duplicate", "training", "future"])
def test_invalid_final_registry_is_rejected(tmp_path, change):
    rows, path = _s13_fixture(tmp_path)
    config = json.loads(path.read_text())
    if change == "duplicate":
        config["finals"].append(config["finals"][0])
    elif change == "training":
        config["feature_eligible"] = True
    else:
        config["finals"][0]["match_date"] = "2026-11-11"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        annotate_reviewed_finals(rows, root=tmp_path)
