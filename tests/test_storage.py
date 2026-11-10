"""Atomic publication fixtures are not competition observations."""

from pathlib import Path

import pandas as pd
import pytest

from mpl_ph_predictor.storage import publish_table, publish_text


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_reader_keeps_old_complete_table_until_new_table_is_ready(tmp_path, monkeypatch, extension):
    target = tmp_path / f"table.{extension}"
    before = pd.DataFrame({"fixture_value": [1]})
    after = pd.DataFrame({"fixture_value": [2]})
    publish_table(before, target)
    read = pd.read_csv if extension == "csv" else pd.read_parquet
    original = getattr(pd.DataFrame, f"to_{extension}")

    def interrupted_write(frame, destination, **kwargs):
        assert Path(destination).parent == target.parent
        assert Path(destination) != target
        Path(destination).write_bytes(b"")
        pd.testing.assert_frame_equal(read(target), before)
        original(frame, destination, **kwargs)
        pd.testing.assert_frame_equal(read(target), before)

    monkeypatch.setattr(pd.DataFrame, f"to_{extension}", interrupted_write)
    publish_table(after, target)
    pd.testing.assert_frame_equal(read(target), after)
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_failed_publication_preserves_old_table_and_cleans_only_its_temporary_file(
    tmp_path, monkeypatch, extension
):
    target = tmp_path / f"table.{extension}"
    target.write_bytes(b"previous fixture bytes")

    def failing_write(frame, destination, **kwargs):
        Path(destination).write_bytes(b"partial fixture bytes")
        raise OSError("fixture disk failure")

    monkeypatch.setattr(pd.DataFrame, f"to_{extension}", failing_write)
    with pytest.raises(OSError, match="fixture disk failure"):
        publish_table(pd.DataFrame({"fixture_value": [1]}), target)
    assert target.read_bytes() == b"previous fixture bytes"
    assert list(tmp_path.iterdir()) == [target]


def test_text_is_published_complete_in_utf8(tmp_path):
    target = tmp_path / "report.json"
    publish_text('{"status": "terverifikasi — sumber tersedia"}', target)
    assert "terverifikasi — sumber tersedia" in target.read_text(encoding="utf-8")
