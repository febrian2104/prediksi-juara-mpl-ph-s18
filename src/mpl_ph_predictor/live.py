"""Save prospective match predictions and evaluate them without rewriting history."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from mpl_ph_predictor.archive import read_snapshot, save_snapshot, sha256, utc_now
from mpl_ph_predictor.data.league_history import load_league_history
from mpl_ph_predictor.features import replay_state
from mpl_ph_predictor.models import predict_probability
from mpl_ph_predictor.simulation import adjust_best_of

ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Semua input dan output harus berada di folder proyek MPL PH.")
    return path


def validate_rules(rules: dict[str, Any], season: int, now: pd.Timestamp) -> None:
    verification = rules.get("verification", {})
    if (
        rules.get("competition") != "MPL Philippines"
        or rules.get("season") != season
        or verification.get("status") != "verified"
        or verification.get("season") != season
        or not verification.get("source_url", "").startswith("https://ph-mpl.com/")
    ):
        raise ValueError("Peserta dan aturan resmi harus diverifikasi untuk musim yang sama.")
    verified_at = pd.Timestamp(verification["verified_at"])
    if verified_at.tzinfo is None or verified_at > now:
        raise ValueError("Waktu verifikasi aturan tidak valid.")
    if not rules.get("participants"):
        raise ValueError("Daftar peserta resmi belum diverifikasi.")


def build_live_predictions(
    schedule: pd.DataFrame,
    history: pd.DataFrame,
    model: Any,
    artifact: dict[str, Any],
    rules: dict[str, Any],
    *,
    now: pd.Timestamp,
) -> pd.DataFrame:
    if schedule.empty:
        raise ValueError("Tidak ada jadwal mendatang; prediksi live tidak dibuat.")
    required = {"match_id", "season", "scheduled_at", "team_a_code", "team_b_code", "best_of"}
    if not required.issubset(schedule):
        raise ValueError(f"Kolom jadwal belum lengkap: {sorted(required - set(schedule))}")
    if now.tzinfo is None:
        raise ValueError("Waktu prediksi harus memiliki zona waktu.")
    future = schedule.copy()
    future["scheduled_at"] = pd.to_datetime(future["scheduled_at"], utc=True, errors="raise")
    if future["scheduled_at"].isna().any() or not future["scheduled_at"].gt(now).all():
        raise ValueError("Semua laga harus belum dimulai pada waktu penyimpanan sebenarnya.")
    if future["match_id"].isna().any() or future["match_id"].duplicated().any():
        raise ValueError("Identitas laga wajib terisi dan unik.")
    if future["season"].isna().any() or future["season"].nunique() != 1:
        raise ValueError("Satu snapshot hanya untuk satu musim.")
    season = int(future["season"].iloc[0])
    validate_rules(rules, season, now)
    teams = set(future["team_a_code"]) | set(future["team_b_code"])
    if not teams.issubset(set(rules["participants"])):
        raise ValueError("Jadwal memuat tim di luar daftar peserta resmi.")
    if future["team_a_code"].eq(future["team_b_code"]).any():
        raise ValueError("Satu tim tidak boleh melawan dirinya sendiri.")
    if not future["best_of"].isin([1, 3, 5, 7]).all():
        raise ValueError("Best-of tidak didukung.")
    forbidden = {"team_a_score", "team_b_score", "winner_team_id", "actual_winner_code"}
    if any(future[column].notna().any() for column in forbidden & set(future)):
        raise ValueError("Jadwal mendatang tidak boleh berisi hasil yang sudah diketahui.")
    if "status" in future and not future["status"].eq("scheduled").all():
        raise ValueError("Status seluruh laga harus scheduled.")

    for column in ("trained_at", "training_latest_match_at", "training_data_observed_at"):
        value = artifact.get(column)
        if not value or pd.Timestamp(value).tzinfo is None or pd.Timestamp(value) > now:
            raise ValueError(f"Provenance waktu model belum valid: {column}.")
    if max(artifact["training_seasons"]) >= season:
        raise ValueError("Model pramusim harus dilatih pada musim yang lebih lama.")
    expected_policy = load_league_history()["rating_policy"]["version"]
    if artifact.get("rating_identity_policy", {}).get("version") != expected_policy:
        raise ValueError("Kebijakan identitas model berbeda; jalankan rebuild sebelum forecast.")

    known = history.copy()
    if "result_observed_at" not in known:
        raise ValueError(
            "Riwayat memerlukan result_observed_at untuk membuktikan ketersediaan hasil."
        )
    observed = pd.to_datetime(known["result_observed_at"], utc=True, errors="coerce")
    starts = pd.to_datetime(known["scheduled_at"], utc=True, errors="raise")
    known = known.loc[observed.le(now) & starts.lt(now) & known["season"].le(season)].copy()
    known = known.loc[known["status"].eq("completed")]
    if known["match_id"].duplicated().any():
        raise ValueError("Riwayat berisi identitas pertandingan ganda.")
    invalid = (
        known[["team_a_score", "team_b_score"]].isna().any(axis=1)
        | known["team_a_score"].eq(known["team_b_score"])
        | known[["team_a_score", "team_b_score"]].lt(0).any(axis=1)
    )
    if invalid.any():
        raise ValueError("Riwayat hasil belum valid.")
    state = replay_state(known, now)
    records = []
    for row in future.itertuples(index=False):
        record = state.feature_record(row.team_a_code, row.team_b_code, season, now)
        x = pd.DataFrame([record])[artifact["feature_columns"]]
        bo3_probability = float(predict_probability(model, x)[0])
        probability = adjust_best_of(bo3_probability, int(row.best_of))
        records.append(
            {
                "match_id": row.match_id,
                "season": season,
                "scheduled_at": row.scheduled_at.isoformat(),
                "team_a_code": row.team_a_code,
                "team_b_code": row.team_b_code,
                "best_of": int(row.best_of),
                "team_a_win_probability": probability,
                "predicted_winner_code": row.team_a_code if probability >= 0.5 else row.team_b_code,
                "model_name": artifact["model_name"],
                "prediction_origin": "saved_pre_match",
                "created_at": now.isoformat(),
                "information_cutoff": now.isoformat(),
                "history_rows_used": len(known),
                "latest_result_observed_at": (
                    str(known["result_observed_at"].max()) if len(known) else None
                ),
            }
        )
    return pd.DataFrame(records)


def save_live_forecast(schedule_path: Path, rules_path: Path, *, period: str) -> Path:
    now = pd.Timestamp(utc_now())
    rules = json.loads(rules_path.read_text())
    model_path = ROOT / "artifacts/final_match_model.joblib"
    artifact = joblib.load(model_path)
    history_path = ROOT / "data/clean/matches.csv"
    schedule = pd.read_csv(schedule_path)
    frame = build_live_predictions(
        schedule, pd.read_csv(history_path), artifact["model"], artifact, rules, now=now
    )
    # Check again just before writing, in case computation crossed a match's start.
    if not pd.to_datetime(frame["scheduled_at"], utc=True).gt(pd.Timestamp(utc_now())).all():
        raise ValueError("Pertandingan sudah dimulai selama perhitungan; snapshot dibatalkan.")
    season = int(frame["season"].iloc[0])
    return save_snapshot(
        {
            "predictions.csv": frame.to_csv(index=False).encode(),
            "rules.json": rules_path.read_bytes(),
            "model.joblib": model_path.read_bytes(),
            "schedule.csv": schedule_path.read_bytes(),
        },
        kind="saved_pre_match",
        metadata={
            "season": season,
            "period": period,
            "prediction_origin": "saved_pre_match",
            "generated_at": now.isoformat(),
            "history_sha256": sha256(history_path.read_bytes()),
            "model_sha256": sha256(model_path.read_bytes()),
            "local_timestamp_note": "Waktu sistem lokal; bukan sertifikasi waktu independen.",
        },
        directory=ROOT / f"data/predictions/live/s{season}",
        key=period,
    )


def evaluate_saved_predictions(
    predictions: pd.DataFrame, results: pd.DataFrame, *, now: pd.Timestamp
) -> pd.DataFrame:
    if set(predictions["prediction_origin"]) != {"saved_pre_match"}:
        raise ValueError("Evaluasi live hanya menerima prediksi saved_pre_match.")
    if results["match_id"].duplicated().any():
        raise ValueError("Hasil pertandingan memuat identitas ganda.")
    joined = predictions.merge(results, on="match_id", how="left", suffixes=("", "_actual"))
    if (
        not pd.to_datetime(joined["created_at"], utc=True)
        .lt(pd.to_datetime(joined["scheduled_at"], utc=True))
        .all()
    ):
        raise ValueError("Prediksi ditemukan dibuat sesudah jadwal laga.")
    observed = pd.to_datetime(joined["result_observed_at"], utc=True, errors="coerce")
    eligible = joined["status"].eq("completed") & observed.le(now)
    invalid_scores = (
        joined[["team_a_score", "team_b_score"]].isna().any(axis=1)
        | joined[["team_a_score", "team_b_score"]].lt(0).any(axis=1)
        | joined["team_a_score"].eq(joined["team_b_score"])
    )
    if (eligible & invalid_scores).any():
        raise ValueError("Skor hasil yang akan dievaluasi belum valid.")
    for column in ("team_a_code", "team_b_code", "season", "scheduled_at"):
        other = f"{column}_actual"
        left, right = joined[column], joined[other]
        if column == "scheduled_at":
            left = pd.to_datetime(left, utc=True)
            right = pd.to_datetime(right, utc=True)
        if (eligible & left.ne(right)).any():
            raise ValueError("Identitas atau jadwal hasil berubah; rekonsiliasi manual diperlukan.")
    if (eligible & observed.le(pd.to_datetime(joined["created_at"], utc=True))).any():
        raise ValueError("Hasil sudah diketahui sebelum prediksi disimpan.")
    eligible &= pd.to_datetime(joined["scheduled_at"], utc=True).lt(now)
    joined["actual_winner_code"] = np.where(
        eligible,
        np.where(
            joined["team_a_score"].gt(joined["team_b_score"]),
            joined["team_a_code"],
            joined["team_b_code"],
        ),
        None,
    )
    joined["evaluation_status"] = np.where(eligible, "evaluated", "pending")
    joined["prediction_correct"] = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    joined.loc[eligible, "prediction_correct"] = joined.loc[eligible, "predicted_winner_code"].eq(
        joined.loc[eligible, "actual_winner_code"]
    )
    return joined


def evaluate_live_snapshot(snapshot_path: Path) -> Path:
    payload, files = read_snapshot(snapshot_path)
    if payload["kind"] != "saved_pre_match":
        raise ValueError("Snapshot bukan prediksi live.")
    frame = evaluate_saved_predictions(
        pd.read_csv(io.BytesIO(files["predictions.csv"])),
        pd.read_csv(ROOT / "data/clean/matches.csv"),
        now=pd.Timestamp(utc_now()),
    )
    evaluated = frame.loc[frame["evaluation_status"].eq("evaluated")]
    metrics: dict[str, Any] = {
        "evaluated_rows": len(evaluated),
        "pending_rows": len(frame) - len(evaluated),
    }
    if len(evaluated):
        from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

        actual = evaluated["actual_winner_code"].eq(evaluated["team_a_code"]).astype(int)
        probabilities = evaluated["team_a_win_probability"]
        metrics.update(
            {
                "accuracy": float(accuracy_score(actual, probabilities.ge(0.5))),
                "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
                "brier_score": float(brier_score_loss(actual, probabilities)),
            }
        )
    return save_snapshot(
        {"evaluation.csv": frame.to_csv(index=False).encode()},
        kind="live_evaluation",
        metadata={
            "prediction_snapshot": str(snapshot_path.relative_to(ROOT)),
            "prediction_sha256": sha256(snapshot_path.read_bytes()),
            **metrics,
        },
        directory=ROOT / "data/predictions/evaluations",
    )
