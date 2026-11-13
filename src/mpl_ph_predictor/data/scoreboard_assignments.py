"""Narrowly reviewed row grouping; never infer rosters or change player statistics."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from mpl_ph_predictor.archive import sha256
from mpl_ph_predictor.data.finals import SOURCE_KINDS, _cached_source

ASSIGNMENT_COLUMNS = [
    "source_team_code",
    "source_canonical_team_code",
    "team_assignment_status",
    "assignment_registry_sha256",
    "assignment_source_urls",
    "assignment_reviewed_on",
]


def load_reviewed_assignments(root: Path, cutoff: str) -> dict[str, dict]:
    """Verify exact reviewed article snapshots offline, retaining failure reasons.

    Missing evidence disables that series' correction, not other valid series.
    A malformed registry aborts before any published data is rewritten.
    """
    path = root / "config/reviewed_scoreboard_assignments.json"
    if not path.is_file():
        return {}
    content = path.read_bytes()
    config = json.loads(content)
    if (
        config["schema_version"] != 1
        or config["feature_eligible"] is not False
        or config["information_cutoff"] != cutoff
    ):
        raise ValueError("Registry penempatan pemain tidak valid atau melampaui scope.")
    entries = config["series"]
    if len({entry["match_id"] for entry in entries}) != len(entries):
        raise ValueError("Seri ganda pada registry penempatan pemain.")
    manifest_path = root / "data/raw/season_evidence_source_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    reviews = {}
    for entry in entries:
        expected_games = entry["team_a_score"] + entry["team_b_score"]
        if not re.fullmatch(r"[a-f0-9]{64}", entry["scoreboard_sha256"]):
            raise ValueError("Review scoreboard memerlukan checksum HTML tertentu.")
        if entry["match_date"] > cutoff or not entry["assignments"] or not entry["sources"]:
            raise ValueError("Review memerlukan tanggal, penempatan dan bukti dalam cutoff.")
        keys = []
        for assignment in entry["assignments"]:
            if (
                assignment["from_team_code"] == assignment["to_team_code"]
                or not assignment["game_numbers"]
                or any(
                    type(game) is not int or game not in range(1, expected_games + 1)
                    for game in assignment["game_numbers"]
                )
            ):
                raise ValueError("Penempatan memerlukan dua tim berbeda dan nomor game valid.")
            keys.extend((game, assignment["player"]) for game in assignment["game_numbers"])
        if len(keys) != len(set(keys)):
            raise ValueError("Penempatan pemain/game ganda dalam review.")
        evidence = []
        for source in entry["sources"]:
            if (
                source["kind"] not in SOURCE_KINDS
                or source["published_on"] > cutoff
                or not source["required_terms"]
                or not re.fullmatch(r"[a-f0-9]{64}", source["reviewed_snapshot_sha256"])
            ):
                raise ValueError("Bukti penempatan harus bertanggal, bertipe dan ber-checksum.")
            record, html = _cached_source(
                root, manifest, source["url"], source["reviewed_snapshot_sha256"]
            )
            if html is not None:
                text = " ".join(BeautifulSoup(html, "html.parser").stripped_strings)
                text = " ".join(text.split()).casefold()
                if any(
                    " ".join(term.split()).casefold() not in text
                    for term in source["required_terms"]
                ):
                    record["status"] = "reviewed_content_missing"
            evidence.append({**record, "kind": source["kind"]})
        reviews[entry["match_id"]] = {
            **entry,
            "registry_sha256": sha256(content),
            "reviewed_on": config["reviewed_on"],
            "evidence": evidence,
            "verified": all(row["status"] == "checksum_verified" for row in evidence),
        }
    return reviews


def verify_review_scope(content: bytes, match: dict, review: dict) -> None:
    if not review["verified"]:
        raise ValueError("Bukti review penempatan pemain hilang/rusak; koreksi ditolak.")
    if sha256(content) != review["scoreboard_sha256"]:
        raise ValueError("Checksum scoreboard berbeda dari review; perlu peninjauan baru.")
    fields = ("match_id", "season", "team_a_code", "team_b_code", "team_a_score", "team_b_score")
    date = pd.Timestamp(match["scheduled_at"]).tz_convert("Asia/Manila").date().isoformat()
    if date != review["match_date"] or any(match[field] != review[field] for field in fields):
        raise ValueError("Fakta seri tidak cocok dengan scope review penempatan pemain.")


def apply_reviewed_assignments(players: list[dict], number: int, names: dict, review: dict) -> None:
    """Move exact existing rows only; caller must still require ten rows and five per team."""
    for assignment in review["assignments"]:
        if number not in assignment["game_numbers"]:
            continue
        candidates = [
            row
            for row in players
            if row["player"] == assignment["player"]
            and row["team_code"] == assignment["from_team_code"]
            and row["source_team_code"] == assignment["from_raw_team_code"]
        ]
        code = assignment["to_team_code"]
        if len(candidates) != 1 or code not in names:
            raise ValueError(
                "Baris/tim asal koreksi tidak cocok atau ambigu; tidak menebak pemain."
            )
        candidates[0].update(
            team=names[code],
            team_code=code,
            team_assignment_status="reviewed_contemporary_report_assignment",
            assignment_registry_sha256=review["registry_sha256"],
            assignment_source_urls=" | ".join(source["url"] for source in review["sources"]),
            assignment_reviewed_on=review["reviewed_on"],
        )
