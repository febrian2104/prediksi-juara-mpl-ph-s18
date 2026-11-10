"""Competition eras and seasonal participants, distinct from ownership/rosters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mpl_ph_predictor.archive import sha256, utc_now

ROOT = Path(__file__).resolve().parents[3]


def load_league_history(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((root / "config/league_history.json").read_text())


def participant_id(season: int, code: str) -> str:
    """A local participation key, never an official franchise/organization ID."""
    return f"ph_s{int(season):02}_{str(code).lower()}"


def competition_regime(season: int, *, start: int = 8) -> str:
    return "pre_franchise" if int(season) < start else "franchise"


def annotate_match_participants(matches: pd.DataFrame, *, root: Path = ROOT) -> pd.DataFrame:
    result = matches.copy()
    history = load_league_history(root)
    start = int(history["franchise_start_season"])
    result["competition_regime"] = result["season"].map(
        lambda season: competition_regime(season, start=start)
    )
    for side in ("a", "b"):
        result[f"team_{side}_participant_id"] = [
            participant_id(season, code)
            for season, code in zip(result["season"], result[f"team_{side}_code"], strict=True)
        ]
        result[f"team_{side}_franchise_slot_id"] = pd.NA
    result["franchise_slot_status"] = result["season"].map(
        lambda season: "not_applicable_pre_franchise" if season < start else "unverified"
    )
    return result


def write_league_identity_audit(*, root: Path = ROOT) -> dict[str, Any]:
    """Publish era metadata plus observed entries/exits without inventing slot chains."""
    history = load_league_history(root)
    start = int(history["franchise_start_season"])
    rows = []
    for season in range(1, 19):
        folder = root / "data" / (f"mpl-season{season}" if season < 18 else "season18")
        teams = pd.read_csv(folder / f"mpl_ph_season_{season:02}_teams.csv")
        for team in teams.itertuples(index=False):
            rows.append(
                {
                    "season": season,
                    "participant_id": participant_id(season, team.team_code),
                    "team_code": team.team_code,
                    "team_name": team.team_name,
                    "competition_regime": competition_regime(season, start=start),
                    "franchise_slot_id": None,
                    "franchise_slot_status": "not_applicable_pre_franchise"
                    if season < start
                    else "unverified",
                    "organization_id": None,
                    "coverage_status": "partial_observed_participants"
                    if season <= 5
                    else "participants_in_collected_matches",
                    "source": team.source,
                }
            )
    participants = pd.DataFrame(rows)
    participants.to_csv(root / "data/clean/season_participants.csv", index=False)
    sources = [*history["verification_sources"]]
    for event in history["events"]:
        sources.extend({"url": url} for url in event["source_urls"])
    manifest = pd.read_csv(root / "data/raw/season_evidence_source_manifest.csv")
    evidence = []
    for url in dict.fromkeys(source["url"] for source in sources):
        candidates = manifest.loc[manifest["url"].eq(url) & manifest["retrieval_status"].eq("ok")]
        status: dict[str, Any] = {"url": url, "status": "missing_cached_evidence"}
        if len(candidates):
            record = candidates.iloc[-1]
            path = (root / record["snapshot_path"]).resolve()
            valid = path.is_relative_to(root.resolve()) and path.is_file()
            valid = valid and sha256(path.read_bytes()) == record["sha256"]
            status.update(
                status="checksum_verified" if valid else "invalid_cached_evidence",
                retrieved_at=record["retrieved_at"],
                snapshot_path=record["snapshot_path"],
                sha256=record["sha256"],
            )
        evidence.append(status)
    changes = []
    for season in range(2, 19):
        prior = set(participants.loc[participants.season.eq(season - 1), "team_code"])
        current = set(participants.loc[participants.season.eq(season), "team_code"])
        changes.append(
            {
                "season": season,
                "observed_new_codes": sorted(current - prior),
                "observed_absent_codes": sorted(prior - current),
                "basis": "collected_participant_difference_not_acquisition_or_slot_evidence",
                "coverage_partial": season <= 6,
            }
        )
    report = {
        "generated_at": utc_now(),
        "franchise_start_season": start,
        "participant_rows": len(participants),
        "unique_participant_ids": int(participants.participant_id.nunique()),
        "rating_policy": history["rating_policy"],
        "source_evidence": evidence,
        "events": history["events"],
        "season_participant_changes": changes,
        "limitation": "Nama, organisasi, slot, roster dan rating berbeda. S1–S5 parsial; "
        "pergantian kode peserta bukan bukti akuisisi. Slot tetap null sampai rantai "
        "kepemilikan diverifikasi; tidak menganggap delapan organisasi selalu sama sejak S8.",
    }
    (root / "reports/league_identity_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report
