from pathlib import Path

import pandas as pd

from mpl_ph_predictor.data.early_results import import_reviewed_early_results

ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_s5_results_never_invent_dates_or_enter_training():
    rows = pd.read_csv(ROOT / "data/mpl-season5/mpl_ph_season_05_matches.csv")
    reviewed = rows.loc[rows.verification_status.eq("reviewed_contemporary_secondary_report")]
    assert len(reviewed) == 8
    assert reviewed[["date", "scheduled_at", "best_of"]].isna().all().all()
    assert reviewed.date_range_start.eq("2020-03-06").all()
    assert reviewed.date_range_end.eq("2020-03-08").all()
    assert not reviewed.training_eligible.any()
    assert not reviewed.match_id.duplicated().any()
    canonical = pd.read_parquet(ROOT / "data/processed/canonical/matches.parquet")
    assert not canonical.match_id.isin(reviewed.match_id).any()


def test_early_import_is_idempotent_and_retains_first_observation(tmp_path):
    import shutil

    for folder in ("data/raw", "reports"):
        (tmp_path / folder).mkdir(parents=True)
    for name in ("reviewed_early_results.json", "early_match_evidence.csv"):
        shutil.copy2(ROOT / "data/raw" / name, tmp_path / "data/raw" / name)
    import_reviewed_early_results(root=tmp_path)
    first = pd.read_csv(tmp_path / "data/raw/early_match_evidence.csv")
    import_reviewed_early_results(root=tmp_path)
    second = pd.read_csv(tmp_path / "data/raw/early_match_evidence.csv")
    pd.testing.assert_frame_equal(first, second)
