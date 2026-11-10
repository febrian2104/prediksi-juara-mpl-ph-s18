"""Date-aware display names without silently merging organization or rating IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def apply_team_name_history(matches: pd.DataFrame, *, root: Path = ROOT) -> pd.DataFrame:
    result = matches.copy()
    timeline = json.loads((root / "config/team_name_history.json").read_text())
    dates = pd.to_datetime(result["scheduled_at"], utc=True).dt.tz_convert("Asia/Manila")
    local_dates = dates.dt.strftime("%Y-%m-%d")
    for side in ("a", "b"):
        result[f"team_{side}_name_basis"] = "identity_catalog_season_review_required"
        result[f"team_{side}_name_source"] = pd.NA
        for transition in timeline["transitions"]:
            selected = result["season"].eq(transition["season"]) & result[f"team_{side}_code"].eq(
                transition["team_code"]
            )
            boundary = transition["announcement_date"]
            if (selected & local_dates.eq(boundary)).any():
                raise ValueError(
                    "Tanggal pergantian nama hanya diketahui harinya; jam perlu audit."
                )
            for mask, name in (
                (selected & local_dates.lt(boundary), transition["name_before"]),
                (selected & local_dates.gt(boundary), transition["name_after"]),
            ):
                result.loc[mask, f"team_{side}_name"] = name
                result.loc[mask, f"team_{side}_name_basis"] = "dated_official_announcement"
                result.loc[mask, f"team_{side}_name_source"] = " | ".join(transition["source_urls"])
    return result
