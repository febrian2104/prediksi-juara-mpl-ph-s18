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


def test_playoff_graph_has_readable_labels_and_winner_loser_paths() -> None:
    import pandas as pd

    from mpl_ph_predictor.dashboard import playoff_bracket_dot

    playoff = pd.read_csv(ROOT / "data/predictions/s18_playoff_predictions.csv")
    graph = playoff_bracket_dot(playoff)
    assert "M3 -> M5" in graph
    assert "M3 -> M6" in graph
    assert "M6 -> M7" in graph
    assert "M6 -> M8" in graph
    assert "Aurora Gaming PH" in graph
    assert r"\u00b7" not in graph


def test_dashboard_refreshes_csv_after_local_update(tmp_path, monkeypatch) -> None:
    import os

    from mpl_ph_predictor import dashboard

    monkeypatch.setattr(dashboard, "PREDICTIONS", tmp_path)
    path = tmp_path / "fixture.csv"
    path.write_text("probability\n0.25\n")
    assert dashboard._csv("fixture.csv").iloc[0]["probability"] == 0.25
    previous = path.stat().st_mtime_ns
    path.write_text("probability\n0.75\n")
    os.utime(path, ns=(previous + 1_000_000, previous + 1_000_000))
    assert dashboard._csv("fixture.csv").iloc[0]["probability"] == 0.75
