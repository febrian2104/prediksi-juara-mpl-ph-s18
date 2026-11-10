from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data" / "predictions"


@st.cache_data(show_spinner=False)
def _cached_csv(path: str, dates: tuple[str, ...], signature: tuple[int, int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _csv(name: str, *, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = PREDICTIONS / name
    if not path.exists():
        raise FileNotFoundError(f"Artefak dashboard tidak ditemukan: {path.name}")
    info = path.stat()
    return _cached_csv(str(path), dates, (info.st_mtime_ns, info.st_size))


def _evaluation() -> dict:
    return json.loads((ROOT / "reports" / "s18_evaluation.json").read_text(encoding="utf-8"))


def _header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
    st.info(
        "Batas informasi kompetisi: 9 November 2026. Prediksi S18 berstatus rekonstruksi, "
        "bukan arsip prediksi live yang disimpan sebelum pertandingan.",
        icon="ℹ️",
    )


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def playoff_bracket_dot(playoff: pd.DataFrame) -> str:
    competition = json.loads((ROOT / "config/competition.json").read_text())
    rows = playoff.set_index("bracket_match_id")
    lines = [
        "digraph bracket { rankdir=LR; bgcolor=transparent;",
        'node [shape=box, style="rounded,filled", fillcolor="#eff6ff", fontname="Arial"];',
        'edge [fontname="Arial", fontsize=10];',
    ]
    for match in competition["playoffs"]["matches"]:
        row = rows.loc[match["id"]]
        label = (
            f"{match['id']} · BO{match['best_of']}\n"
            f"{row.team_a_name}  {row.team_a_score}\n{row.team_b_name}  {row.team_b_score}"
        )
        lines.append(f"{match['id']} [label={json.dumps(label, ensure_ascii=False)}];")
        for reference in (match["team_a"], match["team_b"]):
            if reference.startswith(("winner_", "loser_")):
                outcome, previous = reference.split("_", 1)
                style = 'label="Menang"' if outcome == "winner" else 'label="Kalah", style=dashed'
                lines.append(f"{previous} -> {match['id']} [{style}];")
    lines.append("}")
    return "\n".join(lines)


def page_regular_season() -> None:
    _header(
        "Regular Season",
        "Klasemen akhir, peluang playoff per pekan, dan perbandingan prediksi pertandingan.",
    )
    standings = _csv("s18_standings.csv")
    snapshots = _csv("s18_snapshot_probabilities.csv", dates=("cutoff",))
    matches = _csv("s18_match_predictions.csv", dates=("scheduled_at",))
    regular = matches.loc[matches["stage"].eq("regular_season")].copy()

    st.subheader("Klasemen akhir resmi")
    table = standings[
        [
            "rank",
            "official_name",
            "match_wins",
            "match_losses",
            "game_wins",
            "game_losses",
            "net_game_wins",
        ]
    ].rename(
        columns={
            "rank": "#",
            "official_name": "Tim",
            "match_wins": "Menang",
            "match_losses": "Kalah",
            "game_wins": "Game M",
            "game_losses": "Game K",
            "net_game_wins": "Net Game",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Perkembangan peluang playoff")
    order = ["preseason", *[f"week_{week}" for week in range(1, 9)]]
    plot = snapshots.copy()
    plot["periode"] = pd.Categorical(plot["snapshot_id"], categories=order, ordered=True)
    chart = (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("periode:N", title=None, sort=order),
            y=alt.Y("playoff_probability:Q", title="Peluang playoff", axis=alt.Axis(format="%")),
            color=alt.Color("official_name:N", title="Tim"),
            tooltip=[
                alt.Tooltip("official_name:N", title="Tim"),
                alt.Tooltip("snapshot_id:N", title="Periode"),
                alt.Tooltip("playoff_probability:Q", title="Peluang", format=".1%"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(chart, width="stretch")

    st.subheader("Jadwal, prediksi, dan hasil")
    regular["Minggu"] = regular["week"].astype(int)
    regular["Tanggal (PHT)"] = (
        regular["scheduled_at"].dt.tz_convert("Asia/Manila").dt.strftime("%d %b %Y %H:%M")
    )
    regular["Prediksi"] = regular.apply(
        lambda row: row.team_a_name
        if row.predicted_winner_code == row.team_a_code
        else row.team_b_name,
        axis=1,
    )
    regular["Peluang prediksi"] = regular.apply(
        lambda row: _percent(
            row.team_a_win_probability
            if row.predicted_winner_code == row.team_a_code
            else 1.0 - row.team_a_win_probability
        ),
        axis=1,
    )
    regular["Skor aktual"] = (
        regular["team_a_score"].astype(str) + "–" + regular["team_b_score"].astype(str)
    )
    regular["Status"] = regular["prediction_correct"].map({True: "Tepat", False: "Meleset"})
    regular = regular.sort_values(["Minggu", "scheduled_at"], ascending=[False, False])
    st.dataframe(
        regular[
            [
                "Minggu",
                "Tanggal (PHT)",
                "team_a_name",
                "team_b_name",
                "Prediksi",
                "Peluang prediksi",
                "Skor aktual",
                "Status",
            ]
        ].rename(columns={"team_a_name": "Tim A", "team_b_name": "Tim B"}),
        hide_index=True,
        width="stretch",
    )


def page_playoff() -> None:
    _header(
        "Playoff",
        "Bagan resmi hybrid elimination, peluang juara sebelum playoff, dan hasil aktual.",
    )
    playoff = _csv("s18_playoff_predictions.csv", dates=("scheduled_at",))
    snapshots = _csv("s18_snapshot_probabilities.csv")
    preplayoff = snapshots.loc[snapshots["snapshot_id"].eq("week_8")].sort_values(
        "champion_probability", ascending=False
    )
    st.subheader("Peluang juara sebelum playoff")
    odds = preplayoff[["official_name", "champion_probability"]].copy()
    odds["Peluang juara"] = odds["champion_probability"].map(_percent)
    st.dataframe(
        odds[["official_name", "Peluang juara"]].rename(columns={"official_name": "Tim"}),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Bagan dan hasil")
    st.graphviz_chart(playoff_bracket_dot(playoff), width="stretch")
    st.caption(
        "Bagan menampilkan jalur dan hasil aktual. Peluang di tabel dihitung sebelum playoff "
        "untuk pasangan tim yang akhirnya bertemu; peluang ini bersifat kondisional pada pasangan."
    )
    round_names = {
        "M1": "Play-in: Seed 3 vs 6",
        "M2": "Play-in: Seed 4 vs 5",
        "M3": "Upper Semifinal 1",
        "M4": "Upper Semifinal 2",
        "M5": "Lower Semifinal",
        "M6": "Upper Final",
        "M7": "Lower Final",
        "M8": "Grand Final",
    }
    playoff["Babak"] = playoff["bracket_match_id"].map(round_names)
    playoff["Prediksi"] = playoff.apply(
        lambda row: row.team_a_name
        if row.predicted_winner_code == row.team_a_code
        else row.team_b_name,
        axis=1,
    )
    playoff["Peluang"] = playoff.apply(
        lambda row: _percent(
            row.team_a_win_probability
            if row.predicted_winner_code == row.team_a_code
            else 1 - row.team_a_win_probability
        ),
        axis=1,
    )
    playoff["Skor aktual"] = (
        playoff["team_a_score"].astype(str) + "–" + playoff["team_b_score"].astype(str)
    )
    playoff["Status"] = playoff["prediction_correct"].map({True: "Tepat", False: "Meleset"})
    playoff["Urutan"] = playoff["bracket_match_id"].str.extract(r"(\d+)").astype(int)
    st.dataframe(
        playoff.sort_values("Urutan")[
            [
                "bracket_match_id",
                "Babak",
                "team_a_name",
                "team_b_name",
                "Prediksi",
                "Peluang",
                "Skor aktual",
                "Status",
            ]
        ].rename(
            columns={
                "bracket_match_id": "Match",
                "team_a_name": "Tim A",
                "team_b_name": "Tim B",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.success("Hasil aktual: Aurora Gaming PH juara setelah mengalahkan Team Falcons PH 4–2.")


def page_results() -> None:
    _header(
        "Hasil Prediksi",
        "Ringkasan pilihan juara dan evaluasi probabilitas secara jujur.",
    )
    summary = _csv("s18_champion_summary.csv")
    metrics = _csv("walk_forward_metrics.csv")
    evaluation = _evaluation()
    for row in summary.itertuples(index=False):
        label = "Pramusim" if row.prediction_window == "preseason" else "Sebelum playoff"
        st.subheader(label)
        col1, col2, col3 = st.columns(3)
        col1.metric("Pilihan model", row.predicted_champion_name)
        col2.metric("Peluang", _percent(float(row.predicted_probability)))
        col3.metric("Hasil akhir", row.actual_champion_name)
        st.caption("Status: " + ("tepat" if bool(row.prediction_correct) else "meleset"))

    st.subheader("Evaluasi rekonstruksi S18")
    regular = evaluation["regular_season"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", _percent(regular["accuracy"]))
    col2.metric("Log loss", f"{regular['log_loss']:.3f}")
    col3.metric("Brier score", f"{regular['brier_score']:.3f}")

    st.subheader("Walk-forward backtesting S10–S17")
    display = metrics.copy()
    display["accuracy"] = display["accuracy"].map(_percent)
    for column in ("log_loss", "brier_score", "expected_calibration_error"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    st.dataframe(
        display.rename(
            columns={
                "model": "Model",
                "rows": "Laga uji",
                "accuracy": "Accuracy",
                "log_loss": "Log loss",
                "brier_score": "Brier",
                "expected_calibration_error": "Calibration error",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(evaluation["honesty_note"])
    with st.expander("Kalibrasi probabilitas"):
        predictions = _csv("s18_match_predictions.csv")
        regular_predictions = predictions.loc[predictions["stage"].eq("regular_season")].copy()
        regular_predictions["bin"] = pd.cut(
            regular_predictions["team_a_win_probability"],
            bins=[i / 10 for i in range(11)],
            include_lowest=True,
        )
        calibration = (
            regular_predictions.groupby("bin", observed=True)
            .agg(
                predicted=("team_a_win_probability", "mean"),
                observed=("team_a_win", "mean"),
                rows=("team_a_win", "size"),
            )
            .reset_index(drop=True)
        )
        line = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color="gray", strokeDash=[4, 4])
            .encode(x="x:Q", y="y:Q")
        )
        points = (
            alt.Chart(calibration)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "predicted:Q", title="Probabilitas prediksi", scale=alt.Scale(domain=[0, 1])
                ),
                y=alt.Y(
                    "observed:Q", title="Proporsi kemenangan aktual", scale=alt.Scale(domain=[0, 1])
                ),
                tooltip=[alt.Tooltip("rows:Q", title="Jumlah laga")],
            )
        )
        st.altair_chart(line + points, width="stretch")
        st.caption(
            "Semakin dekat ke garis diagonal, semakin baik kalibrasinya. "
            "Sampel S18 hanya 56 laga regular season."
        )
    with st.expander("Status data Season 1–18 dan riwayat prediksi"):
        st.caption(
            "Sistem franchise PH dimulai S8. Peserta dipisahkan per musim; organisasi, "
            "slot, roster, dan rating tidak disamakan. Akuisisi sesudah S8 tidak otomatis "
            "menggabungkan rating. State pra-franchise direset antarmusim dan pada batas S8; "
            "carryover kode sama setelah S8 tetap merupakan asumsi model."
        )
        coverage = pd.read_csv(ROOT / "reports/season_coverage.csv")
        st.dataframe(
            coverage[
                [
                    "season",
                    "collected_match_rows",
                    "model_eligible_rows",
                    "games_rows",
                    "drafts_rows",
                    "coverage_status",
                ]
            ].rename(
                columns={
                    "season": "Season",
                    "collected_match_rows": "Laga terkumpul",
                    "model_eligible_rows": "Laga layak model",
                    "games_rows": "Game terdata",
                    "drafts_rows": "Observasi hero",
                    "coverage_status": "Status cakupan",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        live_count = len(list((PREDICTIONS / "live").glob("*/*.json")))
        st.caption(
            f"Snapshot pra-laga yang benar-benar disimpan: {live_count}. "
            "S1–S5 masih parsial dan belum masuk model. Roster bertanggal belum lengkap."
        )
        st.caption(
            "Observasi hero berasal dari scoreboard, bukan urutan draft. Ban, urutan pick, "
            "winner per game, dan durasi belum terverifikasi. Detail pascamusim tidak "
            "digunakan sebagai fitur prediksi S18."
        )
        summary_audit_path = ROOT / "reports/game_summary_audit.json"
        if summary_audit_path.is_file():
            game_audit = json.loads(summary_audit_path.read_text())
            st.caption(
                f"Ringkasan dan scoreboard berbeda pada {game_audit['games_with_differences']} "
                f"game ({game_audit['counter_differences']} penghitung). "
                "Nilai sumber dipertahankan; selisih definisi/waktu observasi belum terverifikasi."
            )
        reconciliation_path = ROOT / "reports/season_reconciliation.csv"
        if reconciliation_path.is_file():
            audit = pd.read_csv(reconciliation_path)
            agreed = audit.loc[audit["status"].eq("regular_wl_totals_reconciled"), "season"]
            resolved = audit["status"].isin(
                [
                    "season_wl_totals_reconciled",
                    "mixed_scope_wl_totals_reconciled",
                ]
            )
            review = audit.loc[
                ~resolved & audit["status"].ne("regular_wl_totals_reconciled"), "season"
            ]
            st.caption(
                "Audit silang total W/L regular season cocok pada S"
                + ", S".join(agreed.astype(str))
                + ". Perlu pemeriksaan pada S"
                + ", S".join(review.astype(str))
                + "; kecocokan total bukan bukti semua pertandingan/playoff sudah lengkap."
            )
            st.caption(
                "Cakupan penghitung arsip yang berhasil dijelaskan di luar regular season: "
                + (
                    ", ".join("S" + str(s) for s in audit.loc[resolved, "season"])
                    if resolved.any()
                    else "belum ada"
                )
                + ". Angka ini bukan klasemen regular season dan bukan bukti arsip lengkap."
            )
            st.caption(
                "Statistik pemain arsip merupakan observasi pascamusim: cakupan tahap dan "
                "penetapan tim belum terverifikasi, sehingga tidak dipakai untuk training."
            )
        st.caption(
            "Simulasi tiebreak tiga tim atau lebih masih memakai undian urutan awal "
            "karena total durasi kemenangan belum tersedia. Hasil mingguan direkonstruksi "
            "pada siang hari berikutnya; waktu publikasi hasil historis tidak tersedia."
        )


def main() -> None:
    st.set_page_config(
        page_title="Prediksi MPL PH S18",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    navigation = st.navigation(
        [
            st.Page(page_regular_season, title="Regular Season", icon="📅"),
            st.Page(page_playoff, title="Playoff", icon="🏆"),
            st.Page(page_results, title="Hasil Prediksi", icon="📊"),
        ]
    )
    navigation.run()
