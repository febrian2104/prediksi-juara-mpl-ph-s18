"""Reviewed exact game facts stay separate from untouched scoreboard observations."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mpl_ph_predictor.archive import sha256
from mpl_ph_predictor.data.game_results import BASIS, annotate_game_results, exact_duration_seconds

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, playoff=False):
    (tmp_path / "config").mkdir()
    (tmp_path / "data/raw/source_snapshots").mkdir(parents=True)
    config = json.loads((ROOT / "config/reviewed_game_results.json").read_text())
    config["series"] = [config["series"][-1 if playoff else 0]]
    entry = config["series"][0]
    path = tmp_path / "config/reviewed_game_results.json"
    path.write_text(json.dumps(config))
    shutil.copy2(ROOT / "config/competition.json", tmp_path / "config/competition.json")
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    matches = matches.loc[matches.match_id.eq(entry["match_id"])].copy()
    games = pd.read_csv(ROOT / "data/raw/official_games.csv")
    games = games.loc[games.match_id.eq(entry["match_id"])].copy()
    for name, url, digest in (
        ("game_detail_source_manifest", matches.iloc[0].source_url, entry["scoreboard_sha256"]),
        ("season_evidence_source_manifest", entry["sources"][0]["url"],
         entry["sources"][0]["reviewed_snapshot_sha256"]),
    ):
        manifest = pd.read_csv(ROOT / f"data/raw/{name}.csv")
        selected = manifest.loc[manifest.url.eq(url) & manifest.sha256.eq(digest)].tail(1)
        selected.to_csv(tmp_path / f"data/raw/{name}.csv", index=False)
        relative = selected.iloc[0].snapshot_path
        shutil.copy2(ROOT / relative, tmp_path / relative)
    return games, matches, path


def test_reviewed_game_results_preserve_raw_observations_and_unknowns():
    games = pd.read_csv(ROOT / "data/raw/official_games.csv")
    matches = pd.read_csv(ROOT / "data/clean/matches.csv")
    before = games.copy(deep=True)
    result, report = annotate_game_results(games, matches)
    assert report["status"] == "pass"
    assert report["verified_game_winners"] == 10 and report["verified_exact_durations"] == 7
    assert not report["feature_eligible"]
    assert len(result) == len(games) == 319
    immutable = [c for c in games if c not in {"winner", "duration_minutes", "coverage_status"}]
    pd.testing.assert_frame_equal(result[immutable], before[immutable])
    pd.testing.assert_frame_equal(games, before)
    assert result.winner_source_value.isna().all()
    assert result.duration_minutes_source_value.isna().all()
    assert result.loc[result.season.eq(18), "duration_minutes"].isna().all()
    assert result.loc[result.season.eq(18), "winner"].isna().all()
    assert result.loc[~result.game_result_basis.eq(BASIS).fillna(False), "winner"].isna().all()
    assert result.duration_seconds.dropna().astype(int).tolist() == [
        1131, 691, 789, 737, 761, 1142, 1301,
    ]
    pd.testing.assert_series_equal(
        (result.duration_minutes.dropna() * 60).round().astype("Int64"),
        result.duration_seconds.dropna(), check_names=False,
    )
    assert not result.feature_eligible_for_training.any()
    assert not result.feature_eligible_for_s18.any()
    observed = pd.to_datetime(result.game_result_observed_at.dropna(), utc=True)
    assert observed.gt(pd.Timestamp("2026-11-09", tz="UTC")).all()


@pytest.mark.parametrize(
    "phrase", ["less than 13 minutes", "13 minutes", "12:60-minute match", "0:00-minute match",
               "gold / min"]
)
def test_rounded_invalid_and_ratio_based_durations_are_rejected(phrase):
    with pytest.raises(ValueError):
        exact_duration_seconds(phrase)


@pytest.mark.parametrize(
    "change", ["score", "date", "opponent", "missing_game", "snapshot", "eligible"]
)
def test_mismatching_match_and_scoreboard_scope_fails_closed(tmp_path, change):
    games, matches, _ = fixture(tmp_path)
    if change == "score":
        matches.loc[:, "team_b_score"] = 1
    elif change == "date":
        matches.loc[:, "scheduled_at"] = "2021-08-28T08:00:00Z"
    elif change == "opponent":
        matches.loc[:, "team_b_code"] = "OMG"
    elif change == "missing_game":
        games = games.iloc[:1]
    elif change == "snapshot":
        games.loc[:, "source_snapshot_path"] = "data/raw/source_snapshots/other-test.html"
    else:
        games.loc[:, "feature_eligible_for_training"] = True
    result, report = annotate_game_results(games, matches, root=tmp_path)
    assert report["verified_game_winners"] == report["verified_exact_durations"] == 0
    assert result.winner.isna().all()


@pytest.mark.parametrize(
    "change", ["clock", "rounded", "winner", "missing_winner", "early_clinch", "content"]
)
def test_review_never_infers_missing_facts_or_accepts_an_impossible_sequence(tmp_path, change):
    games, matches, path = fixture(tmp_path, playoff=change == "early_clinch")
    config = json.loads(path.read_text())
    entry = config["series"][0]
    fact = entry["games"][0]
    if change == "clock":
        fact["duration_clock"] = "18:52"  # Does not agree with the pinned quoted 18:51.
    elif change == "rounded":
        fact["duration_evidence"] = "18 minutes"
    elif change == "winner":
        fact["winner_code"] = "ECHO"
    elif change == "missing_winner":
        entry["games"] = entry["games"][:1]
    elif change == "early_clinch":
        entry["games"][1]["winner_code"] = "OMG"
        entry["games"][3]["winner_code"] = "ECHO"
    else:
        fact["evidence_terms"] = ["Unrelated absent test-only assertion"]
    path.write_text(json.dumps(config))
    result, report = annotate_game_results(games, matches, root=tmp_path)
    assert report["status"] == "requires_review" and report["verified_game_winners"] == 0
    assert result.winner.isna().all() and result.duration_minutes.isna().all()


@pytest.mark.parametrize("change", ["missing", "corrupt", "changed_hash"])
def test_report_failure_revokes_prior_enrichment_not_original_observations(tmp_path, change):
    games, matches, _ = fixture(tmp_path)
    prior, report = annotate_game_results(games, matches, root=tmp_path)
    assert report["verified_game_winners"] == 2
    path = tmp_path / "data/raw/season_evidence_source_manifest.csv"
    manifest = pd.read_csv(path)
    snapshot = tmp_path / manifest.iloc[0].snapshot_path
    if change == "missing":
        manifest.loc[manifest.index[0], "snapshot_path"] = "data/raw/source_snapshots/missing.html"
    else:
        content = b"<html>Changed, unreviewed test report.</html>"
        snapshot.write_bytes(content)
        if change == "changed_hash":
            manifest.loc[manifest.index[0], "sha256"] = sha256(content)
    manifest.to_csv(path, index=False)
    result, report = annotate_game_results(prior, matches, root=tmp_path)
    assert report["verified_game_winners"] == 0
    assert result.winner.isna().all() and result.duration_minutes.isna().all()
    assert result.game_result_registry_sha256.isna().all()
    assert result.coverage_status.tolist() == games.coverage_status.tolist()


@pytest.mark.parametrize("change", ["training", "cutoff", "duplicate", "future"])
def test_invalid_registry_aborts_before_export(tmp_path, change):
    games, matches, path = fixture(tmp_path)
    config = json.loads(path.read_text())
    if change == "training":
        config["feature_eligible"] = True
    elif change == "cutoff":
        config["information_cutoff"] = "2026-11-13"
    elif change == "duplicate":
        config["series"].append(config["series"][0])
    else:
        config["series"][0]["sources"][0]["published_on"] = "2026-11-14"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        annotate_game_results(games, matches, root=tmp_path)


def test_newer_report_revision_does_not_replace_reviewed_bytes(tmp_path):
    games, matches, _ = fixture(tmp_path)
    path = tmp_path / "data/raw/season_evidence_source_manifest.csv"
    manifest = pd.read_csv(path)
    newer = manifest.iloc[0].to_dict()
    content = b"<html>Later unreviewed test report.</html>"
    newer.update(sha256=sha256(content), snapshot_path="data/raw/source_snapshots/new-test.html")
    (tmp_path / newer["snapshot_path"]).write_bytes(content)
    pd.concat([manifest, pd.DataFrame([newer])], ignore_index=True).to_csv(path, index=False)
    _, report = annotate_game_results(games, matches, root=tmp_path)
    assert report["status"] == "pass"
    assert report["series"][0]["sources"][0]["sha256"] == manifest.iloc[0].sha256


def test_existing_source_conflict_is_not_silently_overwritten(tmp_path):
    games, matches, _ = fixture(tmp_path)
    games["winner"] = games.winner.astype("string")
    games.loc[games.index[0], "winner"] = "team_b"
    result, report = annotate_game_results(games, matches, root=tmp_path)
    assert report["verified_game_winners"] == 0
    assert result.loc[games.index[0], "winner"] == "team_b"
