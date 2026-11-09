from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_entrypoint_and_precomputed_artifacts_exist() -> None:
    assert (ROOT / "streamlit_app.py").exists()
    required = {
        "s18_standings.csv",
        "s18_match_predictions.csv",
        "s18_snapshot_probabilities.csv",
        "s18_playoff_predictions.csv",
        "s18_champion_summary.csv",
        "walk_forward_metrics.csv",
    }
    available = {path.name for path in (ROOT / "data" / "predictions").glob("*.csv")}
    assert required.issubset(available)


def test_dashboard_has_exactly_three_named_pages() -> None:
    source = (ROOT / "src" / "mpl_ph_predictor" / "dashboard.py").read_text()
    assert source.count("st.Page(") == 3
    assert 'title="Regular Season"' in source
    assert 'title="Playoff"' in source
    assert 'title="Hasil Prediksi"' in source
