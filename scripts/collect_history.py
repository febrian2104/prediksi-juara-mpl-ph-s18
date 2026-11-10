"""Audit availability of early MPL PH sources without fabricating missing matches."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpl_ph_predictor.archive import archive_project_state  # noqa: E402
from mpl_ph_predictor.data import build_canonical_data  # noqa: E402
from mpl_ph_predictor.data.catalog import export_reference_layout  # noqa: E402
from mpl_ph_predictor.data.early_results import import_reviewed_early_results  # noqa: E402
from mpl_ph_predictor.data.league_history import write_league_identity_audit  # noqa: E402
from mpl_ph_predictor.data.research import collect_historical_source_availability  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-only", action="store_true", help="Ekspor bukti lokal tanpa web.")
    args = parser.parse_args()
    archive_project_state(reason="before_early_evidence_update")
    if not args.cache_only:
        frame = collect_historical_source_availability()
        print(frame[["season", "url", "http_status", "retrieval_status"]].to_string(index=False))
    report = import_reviewed_early_results()
    matches, teams, rosters, quality = build_canonical_data()
    if quality["status"] != "pass":
        raise RuntimeError("Data canonical gagal audit; ekspor dibatalkan.")
    export_reference_layout(matches, teams, rosters)
    write_league_identity_audit()
    archive_project_state(reason="after_early_evidence_update")
    print(
        json.dumps(
            {
                "reviewed_result_rows": report["reviewed_result_rows"],
                "total_early_result_rows": report["total_early_result_rows"],
                "report": "reports/early_results_audit.json",
                "network_used": not args.cache_only,
                "training_performed": False,
                "prediction_files_changed": False,
            },
            indent=2,
        )
    )
