import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mpl_ph_predictor.data.catalog import championship_table, match_table, team_table
from mpl_ph_predictor.data.early_results import audit_early_coverage, import_reviewed_early_results

ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_s5_results_never_invent_dates_or_enter_training():
    rows = pd.read_csv(ROOT / "data/mpl-season5/mpl_ph_season_05_matches.csv")
    reviewed = rows.loc[rows.verification_status.eq("reviewed_contemporary_secondary_report")]
    assert len(rows) == 51
    assert len(reviewed) == 50
    weekly = reviewed.loc[reviewed.stage.eq("regular_season") & reviewed.week.ge(2)]
    assert len(weekly) == 40
    assert weekly["date"].isna().all()
    assert reviewed[["scheduled_at", "best_of"]].isna().all().all()
    w2 = weekly.loc[weekly.week.eq(2)]
    assert w2.date_range_start.eq("2020-03-06").all()
    assert w2.date_range_end.eq("2020-03-08").all()
    assert set(weekly.week) == {2, 3, 4, 5, 6}
    playoff = reviewed.loc[reviewed.stage.eq("playoffs")]
    assert len(playoff) == 9
    assert playoff["date"].eq(playoff.date_range_start).all()
    assert not reviewed.training_eligible.any()
    assert not reviewed.match_id.duplicated().any()
    canonical = pd.read_parquet(ROOT / "data/processed/canonical/matches.parquet")
    assert not canonical.match_id.isin(reviewed.match_id).any()


def _copy_evidence(tmp_path):
    for folder in ("data/raw", "reports"):
        (tmp_path / folder).mkdir(parents=True)
    paths = [
        *(ROOT / "data/raw").glob("reviewed_early_results*.json"),
        ROOT / "data/raw/early_match_evidence.csv",
    ]
    for path in paths:
        shutil.copy2(path, tmp_path / "data/raw" / path.name)


def test_early_import_is_idempotent_and_retains_first_observation(tmp_path):
    _copy_evidence(tmp_path)
    import_reviewed_early_results(root=tmp_path)
    first = pd.read_csv(tmp_path / "data/raw/early_match_evidence.csv")
    import_reviewed_early_results(root=tmp_path)
    second = pd.read_csv(tmp_path / "data/raw/early_match_evidence.csv")
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    "change",
    ["conflict", "reverse_duplicate", "negative", "boolean", "same_team", "training", "date_range"],
)
def test_invalid_or_conflicting_review_never_overwrites_old_results(tmp_path, change):
    _copy_evidence(tmp_path)
    path = tmp_path / "data/raw/reviewed_early_results.json"
    evidence = json.loads(path.read_text())
    result = evidence["results"][0]
    if change == "conflict":
        result["team_a_score"] = 1
    elif change == "reverse_duplicate":
        evidence["results"].append(
            {
                "team_a_code": result["team_b_code"],
                "team_a_name": result["team_b_name"],
                "team_b_code": result["team_a_code"],
                "team_b_name": result["team_a_name"],
                "team_a_score": result["team_b_score"],
                "team_b_score": result["team_a_score"],
            }
        )
    elif change == "negative":
        result["team_a_score"] = -1
    elif change == "boolean":
        result["team_a_score"] = True
    elif change == "same_team":
        result["team_b_code"] = result["team_a_code"]
    elif change == "training":
        evidence["training_eligible"] = True
    else:
        evidence["date_range_end"] = "2020-03-01"
    path.write_text(json.dumps(evidence))
    raw = tmp_path / "data/raw/early_match_evidence.csv"
    before = raw.read_bytes()
    with pytest.raises(ValueError):
        import_reviewed_early_results(root=tmp_path)
    assert raw.read_bytes() == before


def test_s5_missing_pairs_and_standings_conflicts_are_not_hidden():
    rows = pd.read_csv(ROOT / "data/raw/early_match_evidence.csv")
    before = rows.copy(deep=True)
    report = audit_early_coverage(rows)[0]
    assert report["regular_rows"] == 41
    assert report["expected_regular_rows"] == 45
    assert report["playoff_rows"] == 10
    assert report["missing_regular_pairs"] == [
        ["BLCK", "STI"],
        ["BSB", "SGD"],
        ["EXE", "ONIC"],
        ["GF", "ULVL"],
    ]
    assert not report["duplicate_regular_pairs"]
    assert not report["training_eligible"]
    assert report["standings_comparison"]["archive_wl_scope"] == "unresolved"
    pd.testing.assert_frame_equal(rows, before)


def test_finalists_come_from_explicit_grand_final_not_import_order():
    raw = pd.read_csv(ROOT / "data/raw/early_match_evidence.csv")
    matches = match_table(raw.loc[raw.season.eq(5)])
    teams = team_table(matches)
    result = championship_table(matches, teams).set_index("team")
    assert result.loc["Sunsparks", "champion"] == 1
    assert result.loc["ONIC Philippines", "runner_up"] == 1
    no_final = matches.loc[~matches.stage.eq("grand_final")]
    unknown = championship_table(no_final, teams)
    assert unknown.champion.isna().all()
    assert unknown.final_rank.isna().all()
    assert unknown.rank_coverage.eq("grand_final_not_explicitly_identified").all()


def test_reviewed_sources_have_matching_local_evidence_checksums():
    from mpl_ph_predictor.archive import sha256

    raw = pd.read_csv(ROOT / "data/raw/early_match_evidence.csv")
    reviewed = raw.loc[raw.evidence_record_path.notna()]
    for path, group in reviewed.groupby("evidence_record_path"):
        assert group.evidence_record_sha256.eq(sha256((ROOT / path).read_bytes())).all()
