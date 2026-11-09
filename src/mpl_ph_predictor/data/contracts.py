"""Portable reference column contracts; no dependency on the Indonesia workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def load_dataset_schema() -> dict[str, Any]:
    return json.loads((ROOT / "config/dataset_schema.json").read_text())


def conform_table(frame: pd.DataFrame, table: str, *, live: bool = False) -> pd.DataFrame:
    schema = load_dataset_schema()
    columns = schema["live_tables" if live else "tables"][table]
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    for column in set(columns) & set(schema["integer_columns"]):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("Int64")
    for column in set(columns) & set(schema["number_columns"]):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("Float64")
    numeric = set(schema["integer_columns"]) | set(schema["number_columns"])
    for column in set(columns) - numeric:
        result[column] = result[column].astype("string")
    return result[columns + [column for column in result if column not in columns]]


def variable_coverage(frame: pd.DataFrame, table: str, season: int) -> list[dict[str, Any]]:
    columns = load_dataset_schema()["tables"][table]
    records = []
    for column in columns:
        available = int(frame[column].notna().sum())
        records.append(
            {
                "season": season,
                "table": table,
                "variable": column,
                "rows": len(frame),
                "non_null_rows": available,
                "availability": "unavailable"
                if available == 0
                else "complete_for_collected_rows"
                if available == len(frame)
                else "partial",
            }
        )
    return records
