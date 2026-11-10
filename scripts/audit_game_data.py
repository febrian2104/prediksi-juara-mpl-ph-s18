"""Audit cached game summary counters and export quality flags without training."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpl_ph_predictor.archive import archive_project_state  # noqa: E402
from mpl_ph_predictor.data.catalog import export_reference_layout  # noqa: E402
from mpl_ph_predictor.data.game_audit import audit_game_summaries  # noqa: E402
from mpl_ph_predictor.data.normalization import build_canonical_data  # noqa: E402

if __name__ == "__main__":
    archive_project_state(reason="before_game_summary_audit")
    report = audit_game_summaries()
    matches, teams, rosters, quality = build_canonical_data()
    if quality["status"] != "pass":
        raise RuntimeError("Canonical dataset gagal audit; ekspor dibatalkan.")
    export_reference_layout(matches, teams, rosters)
    archive_project_state(reason="after_game_summary_audit")
    print(json.dumps(report, indent=2, ensure_ascii=False))
