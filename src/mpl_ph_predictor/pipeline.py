from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from mpl_ph_predictor.archive import archive_project_state, verify_history
from mpl_ph_predictor.collector import collect_official_data, write_raw_data
from mpl_ph_predictor.data import build_canonical_data, write_canonical_data
from mpl_ph_predictor.data.catalog import export_processed_outputs, export_reference_layout
from mpl_ph_predictor.data.game_details import collect_game_details
from mpl_ph_predictor.data.research import collect_historical_source_availability
from mpl_ph_predictor.data.season_evidence import collect_season_evidence
from mpl_ph_predictor.features import build_match_features, write_features
from mpl_ph_predictor.models import (
    fit_production_model,
    walk_forward_compare,
    write_model_outputs,
)
from mpl_ph_predictor.simulation import build_s18_outputs, write_s18_outputs

ROOT = Path(__file__).resolve().parents[2]


def _evaluate_s18(predictions: pd.DataFrame) -> dict[str, Any]:
    actual = predictions["team_a_win"].astype(int)
    probabilities = predictions["team_a_win_probability"].astype(float)
    regular = predictions.loc[predictions["stage"].eq("regular_season")]
    return {
        "scope": "reconstructed_pre_match_predictions",
        "honesty_note": (
            "Prediksi dibangun ulang saat pipeline dijalankan dengan state sebelum laga; "
            "bukan arsip prediksi yang benar-benar disimpan pada tanggal pertandingan."
        ),
        "all_s18": {
            "rows": int(len(predictions)),
            "accuracy": float(accuracy_score(actual, probabilities.ge(0.5))),
            "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
            "brier_score": float(brier_score_loss(actual, probabilities)),
        },
        "regular_season": {
            "rows": int(len(regular)),
            "accuracy": float(
                accuracy_score(regular["team_a_win"], regular["team_a_win_probability"].ge(0.5))
            ),
            "log_loss": float(
                log_loss(
                    regular["team_a_win"],
                    regular["team_a_win_probability"],
                    labels=[0, 1],
                )
            ),
            "brier_score": float(
                brier_score_loss(regular["team_a_win"], regular["team_a_win_probability"])
            ),
        },
    }


def run_pipeline(*, fetch: bool = True) -> dict[str, Any]:
    before = archive_project_state(reason="before_pipeline_update")
    if fetch:
        write_raw_data(collect_official_data())
        collect_historical_source_availability()
    required = ROOT / "data" / "raw" / "official_matches.csv"
    if not required.exists():
        raise FileNotFoundError("Data mentah belum ada; jalankan pipeline tanpa --no-fetch.")

    matches, teams, rosters, quality = build_canonical_data()
    write_canonical_data(matches, teams, rosters, quality)
    if quality["status"] != "pass":
        raise RuntimeError(f"Audit canonical dataset gagal: {quality}")
    if fetch:
        collect_season_evidence()
        collect_game_details()
    coverage = export_reference_layout(matches, teams, rosters)

    features = build_match_features(matches)
    feature_report = write_features(features)
    if feature_report["history_after_match_count"]:
        raise RuntimeError("Audit leakage fitur gagal.")

    metrics, backtest_predictions = walk_forward_compare(features)
    _, model, artifact = fit_production_model(features, metrics)
    write_model_outputs(metrics, backtest_predictions, model, artifact)
    outputs = build_s18_outputs(matches, features, model, artifact)
    write_s18_outputs(outputs)
    export_processed_outputs()
    evaluation = _evaluate_s18(outputs["s18_match_predictions"])
    (ROOT / "reports" / "s18_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "season_coverage": coverage.to_dict(orient="records"),
        "previous_snapshot": str(before.relative_to(ROOT)) if before else None,
        "data_quality": quality,
        "feature_audit": feature_report,
        "production_model": artifact,
        "walk_forward_metrics": metrics.to_dict(orient="records"),
        "s18_evaluation": evaluation,
    }
    (ROOT / "reports" / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    archive_project_state(reason="after_pipeline_update")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Perbarui data dan prediksi MPL PH.")
    parser.add_argument(
        "--no-fetch", action="store_true", help="Gunakan data mentah lokal tanpa mengakses web."
    )
    parser.add_argument("--archive-only", action="store_true", help="Arsipkan artefak saat ini.")
    parser.add_argument("--verify-history", action="store_true", help="Periksa integritas arsip.")
    args = parser.parse_args()
    if args.archive_only:
        print(archive_project_state(reason="manual_backup"))
        return
    if args.verify_history:
        print(json.dumps(verify_history(), indent=2))
        return
    summary = run_pipeline(fetch=not args.no_fetch)
    print(
        json.dumps(
            {
                "status": summary["data_quality"]["status"],
                "canonical_matches": summary["data_quality"]["canonical_match_rows"],
                "model": summary["production_model"]["model_name"],
                "season_coverage_report": "reports/season_coverage.csv",
                "full_report": "reports/pipeline_summary.json",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
