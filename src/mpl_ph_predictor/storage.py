"""Publish complete tables atomically while keeping temporary files beside their target."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def _staged_output(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        # Only this operation's exact temporary file is removed; old data is untouched.
        temporary.unlink(missing_ok=True)


def publish_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix not in {".csv", ".parquet"}:
        raise ValueError("Format tabel ekspor harus CSV atau Parquet.")
    with _staged_output(path) as temporary:
        if path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            frame.to_parquet(temporary, index=False)


def publish_text(content: str, path: Path) -> None:
    with _staged_output(path) as temporary:
        temporary.write_text(content, encoding="utf-8")
