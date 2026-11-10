"""Attach reviewed grand-final markers; never infer finals from row order."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from mpl_ph_predictor.archive import sha256, utc_now
from mpl_ph_predictor.data.collector import _parse_historical_page, _parse_schedule

ROOT = Path(__file__).resolve().parents[3]
BASIS = "reviewed_final_registry_with_checksummed_sources"
SOURCE_KINDS = {
    "league_hosted_syndicated_report",
    "league_hosted_syndicated_excerpt",
    "league_hosted_match_report",
    "secondary_original_of_league_linked_report",
    "secondary_contemporary_report",
    "official_publisher_release",
}


def _cached_source(
    root: Path, manifest: pd.DataFrame, url: str, reviewed_sha256: str | None = None
) -> tuple[dict, bytes | None]:
    evidence: dict[str, Any] = {"url": url, "status": "missing_cached_evidence"}
    if manifest.empty:
        return evidence, None
    candidates = manifest.loc[manifest.url.eq(url)]
    if reviewed_sha256 is not None:
        # Later retrievals may contain changed facts or page chrome. The human review
        # binds to exact bytes, not whichever successful download happens to be last.
        candidates = candidates.loc[candidates.sha256.eq(reviewed_sha256)]
    if "retrieval_status" in candidates:
        candidates = candidates.loc[candidates.retrieval_status.eq("ok")]
    if candidates.empty:
        return evidence, None
    row = candidates.iloc[-1]
    relative, digest = row.get("snapshot_path"), row.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        return {**evidence, "status": "invalid_cached_evidence"}, None
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return {**evidence, "status": "invalid_cached_evidence"}, None
    content = path.read_bytes()
    if sha256(content) != digest:
        return {**evidence, "status": "invalid_cached_evidence"}, None
    return {
        **evidence,
        "status": "checksum_verified",
        "snapshot_path": relative,
        "sha256": digest,
        "retrieved_at": row["retrieved_at"],
    }, content


def _date(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("Asia/Manila").date().isoformat()


def _verify_card(row: pd.Series, expected: dict, manifest: pd.DataFrame, root: Path) -> dict:
    season = expected["season"]
    # S8's canonical source points to a detail page; its result card is in /schedule.
    url = f"https://ph-mpl.com/s{season}" + ("/schedule" if season == 8 else "")
    evidence, content = _cached_source(root, manifest, url)
    if content is None:
        return evidence
    soup = BeautifulSoup(content, "html.parser")
    try:
        cards = (
            _parse_schedule(soup, url, season=season)
            if season == 8
            else _parse_historical_page(soup, season, url, [])
        )
        selected = [card for card in cards if card["official_match_id"] == row.official_match_id]
        if len(selected) != 1:
            return {**evidence, "status": "official_card_missing_or_ambiguous"}
        card = selected[0]
        fields = ("season", "stage", "team_a_code", "team_b_code", "team_a_score", "team_b_score")
        conflicts = [
            field for field in fields if card[field] != row.get(f"raw_{field}", row[field])
        ]
        if _date(card["scheduled_at"]) != expected["match_date"]:
            conflicts.append("match_date")
        if pd.Timestamp(card["scheduled_at"]) != pd.Timestamp(row.scheduled_at):
            conflicts.append("scheduled_at")
        return {
            **evidence,
            "status": "official_card_verified" if not conflicts else "official_card_conflict",
            "conflicting_fields": conflicts,
            "team_code_comparison": "official_raw_codes_before_season_scoped_aliases",
        }
    except (ValueError, KeyError, TypeError) as error:
        return {**evidence, "status": "official_card_parse_failed", "error": str(error)}


def annotate_reviewed_finals(
    matches: pd.DataFrame, *, root: Path = ROOT
) -> tuple[pd.DataFrame, dict]:
    """Verify registry, match facts, news content and official cards without a network call.

    These annotations are retrospective result metadata, not pre-match features.
    An unavailable/corrupt source fails closed, leaving the final marker unknown.
    """
    result = matches.copy()
    prior = result.get("grand_final_basis", pd.Series(index=result.index, dtype="string")).eq(BASIS)
    for column in (
        "round",
        "grand_final_basis",
        "grand_final_source_urls",
        "grand_final_verification_status",
        "grand_final_reviewed_on",
        "grand_final_registry_sha256",
    ):
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
        result.loc[prior.fillna(False), column] = pd.NA
    path = root / "config/reviewed_grand_finals.json"
    report = {
        "generated_at": utc_now(),
        "status": "registry_not_available",
        "scope": "explicit_final_identity_not_complete_bracket",
        "finals": [],
    }
    if not path.is_file():
        return result, report
    content = path.read_bytes()
    config = json.loads(content)
    if config["schema_version"] != 1 or config["feature_eligible"] is not False:
        raise ValueError("Registry final tidak valid atau tidak boleh menjadi fitur pra-laga.")
    entries = config["finals"]
    if len({entry["season"] for entry in entries}) != len(entries) or len(
        {entry["match_id"] for entry in entries}
    ) != len(entries):
        raise ValueError("Registry berisi identitas final ganda.")
    for entry in entries:
        if not entry["sources"] or any(
            source["kind"] not in SOURCE_KINDS for source in entry["sources"]
        ):
            raise ValueError("Final memerlukan jenis sumber yang dikenali.")
        if any(
            not re.fullmatch(r"[a-f0-9]{64}", source.get("reviewed_snapshot_sha256", ""))
            or not source["required_terms"]
            for source in entry["sources"]
        ):
            raise ValueError("Bukti review final memerlukan checksum dan penanda isi.")
        if entry["match_date"] > config["information_cutoff"]:
            raise ValueError("Final melampaui cutoff informasi.")
    manifests = {}
    for name in ("source_manifest", "season_evidence_source_manifest"):
        manifest_path = root / f"data/raw/{name}.csv"
        manifests[name] = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    audits = []
    for entry in entries:
        if entry["season"] not in set(result.season):
            continue
        audit = {
            "season": entry["season"],
            "match_id": entry["match_id"],
            "status": "match_missing_or_ambiguous",
            "article_sources": [],
            "article_score_verified": entry["article_score_verified"],
        }
        audits.append(audit)
        candidates = result.loc[result.match_id.eq(entry["match_id"])]
        if len(candidates) != 1:
            continue
        row = candidates.iloc[0]
        fields = ("season", "team_a_code", "team_b_code", "team_a_score", "team_b_score")
        conflicts = [field for field in fields if row[field] != entry[field]]
        if row.stage != "playoffs" or _date(row.scheduled_at) != entry["match_date"]:
            conflicts.append("stage_or_match_date")
        winner = row.team_a_code if row.team_a_score > row.team_b_score else row.team_b_code
        if row.team_a_score == row.team_b_score or winner != entry["winner_code"]:
            conflicts.append("winner_code")
        if conflicts:
            audit.update(status="match_facts_conflict", conflicting_fields=conflicts)
            continue
        audit["official_card"] = _verify_card(row, entry, manifests["source_manifest"], root)
        for source in entry["sources"]:
            evidence, html = _cached_source(
                root,
                manifests["season_evidence_source_manifest"],
                source["url"],
                reviewed_sha256=source["reviewed_snapshot_sha256"],
            )
            evidence["kind"] = source["kind"]
            if html is not None:
                text = " ".join(BeautifulSoup(html, "html.parser").stripped_strings).casefold()
                missing = [
                    term
                    for term in source["required_terms"]
                    if " ".join(term.split()).casefold() not in " ".join(text.split())
                ]
                if missing:
                    evidence.update(status="reviewed_content_missing", missing_terms=missing)
            audit["article_sources"].append(evidence)
        valid = audit["official_card"]["status"] == "official_card_verified" and all(
            source["status"] == "checksum_verified" for source in audit["article_sources"]
        )
        audit["status"] = "verified" if valid else "source_evidence_requires_review"
        if valid:
            urls = " | ".join(source["url"] for source in entry["sources"])
            values = {
                "round": "grand_final",
                "grand_final_basis": BASIS,
                "grand_final_source_urls": urls,
                "grand_final_verification_status": "reviewed_sources_and_official_card_verified",
                "grand_final_reviewed_on": config["reviewed_on"],
                "grand_final_registry_sha256": sha256(content),
            }
            for column, value in values.items():
                result.loc[candidates.index, column] = value
    report.update(
        status="pass" if all(row["status"] == "verified" for row in audits) else "requires_review",
        reviewed_on=config["reviewed_on"],
        information_cutoff=config["information_cutoff"],
        registry_sha256=sha256(content),
        verified_final_rows=sum(row["status"] == "verified" for row in audits),
        finals=audits,
        feature_eligible=False,
    )
    return result, report
