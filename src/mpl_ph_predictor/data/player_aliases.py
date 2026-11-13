"""Reviewed, season/team-scoped IGN links; never infer global player identities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
from bs4 import BeautifulSoup

from mpl_ph_predictor.archive import sha256, utc_now
from mpl_ph_predictor.data.finals import _cached_source

ROOT = Path(__file__).resolve().parents[3]
STATUS = "reviewed_season_team_alias"
METADATA_COLUMNS = [
    "player_alias_basis",
    "player_alias_registry_sha256",
    "player_alias_source_urls",
    "player_alias_reviewed_on",
    "player_link_scope",
]
BASES = {
    "reviewed_official_roster_asset_alias",
    "reviewed_contemporary_reports_same_full_name",
}


def _manifest(root: Path, name: str) -> pd.DataFrame:
    path = root / f"data/raw/{name}.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _digest(value: str) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{64}", value))


def annotate_player_aliases(
    statistics: pd.DataFrame, players: pd.DataFrame, *, root: Path = ROOT
) -> tuple[pd.DataFrame, dict]:
    """Apply only manually reviewed links with exact, checksummed source evidence.

    Original IGN, team, numeric observations and temporal eligibility are untouched.
    Unknown/corrupt evidence leaves an alias unresolved. No fuzzy matching is used.
    """
    result = statistics.copy()
    prior = result["player_identity_status"].eq(STATUS)
    result.loc[prior, "player_identity_status"] = "unresolved_ign_alias"
    result.loc[prior, ["player_id", "matched_roster_player", "role"]] = pd.NA
    for column in METADATA_COLUMNS:
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
        result.loc[prior, column] = pd.NA
    report = {
        "generated_at": utc_now(),
        "status": "registry_not_available",
        "scope": "season_team_observation_not_global_person_or_effective_roster",
        "feature_eligible": False,
        "aliases": [],
    }
    path = root / "config/reviewed_player_aliases.json"
    if not path.is_file():
        return result, report
    content = path.read_bytes()
    config = json.loads(content)
    cutoff = json.loads((root / "config/competition.json").read_text())["data_cutoff"]
    if (
        config["schema_version"] != 1
        or config["feature_eligible"] is not False
        or config["information_cutoff"] != cutoff
        or not _digest(config["statistics_source"]["reviewed_snapshot_sha256"])
    ):
        raise ValueError("Registry alias tidak valid, melampaui cutoff, atau mengaktifkan fitur.")
    entries = config["aliases"]
    keys = [(e["season"], e["team_code"], e["statistics_ign"]) for e in entries]
    targets = [(e["season"], e["team_code"], e["roster_ign"]) for e in entries]
    if len(keys) != len(set(keys)) or len(targets) != len(set(targets)):
        raise ValueError("Alias atau target ganda; tidak menggabungkan observasi statistik.")
    for entry in entries:
        if (
            entry["basis"] not in BASES
            or not _digest(entry["roster_source"]["reviewed_snapshot_sha256"])
            or entry["statistics_ign"] == entry["roster_ign"]
            or not entry["statistics_ign"].strip()
            or not entry["roster_ign"].strip()
        ):
            raise ValueError("Review alias memerlukan dua IGN, jenis bukti dan checksum valid.")
        if entry["basis"] == "reviewed_contemporary_reports_same_full_name":
            if len(entry["supporting_sources"]) < 2:
                raise ValueError("Pencocokan nama lengkap memerlukan dua laporan bertanggal.")
        for source in entry["supporting_sources"]:
            if (
                source["kind"] != "secondary_contemporary_report"
                or pd.Timestamp(source["published_on"]) > pd.Timestamp(cutoff)
                or not source["required_terms"]
                or not _digest(source["reviewed_snapshot_sha256"])
            ):
                raise ValueError("Sumber alias harus bertanggal dalam cutoff dan ber-checksum.")
    official = _manifest(root, "source_manifest")
    supporting = _manifest(root, "season_evidence_source_manifest")
    stat_source = config["statistics_source"]
    stat_evidence, _ = _cached_source(
        root, official, stat_source["url"], stat_source["reviewed_snapshot_sha256"]
    )
    audits = []
    for entry in entries:
        audit = {
            "season": entry["season"],
            "team_code": entry["team_code"],
            "statistics_ign": entry["statistics_ign"],
            "roster_ign": entry["roster_ign"],
            "basis": entry["basis"],
            "status": "requires_review",
            "sources": [stat_evidence],
        }
        audits.append(audit)
        try:
            if stat_evidence["status"] != "checksum_verified":
                raise ValueError("Snapshot statistik yang ditinjau tidak tersedia/valid.")
            selected = result.loc[
                result.season.eq(entry["season"])
                & result.team_code.eq(entry["team_code"])
                & result.player.eq(entry["statistics_ign"])
            ]
            target = players.loc[
                players.season.eq(entry["season"])
                & players.team_code.eq(entry["team_code"])
                & players.player.eq(entry["roster_ign"])
                & players.member_type.eq("player")
            ]
            if len(selected) != 1 or len(target) != 1:
                raise ValueError("Observasi/target roster tidak tunggal dalam tim dan musim.")
            observed, roster = selected.iloc[0], target.iloc[0]
            if (
                observed.source != stat_source["url"]
                or observed.source_snapshot_sha256 != stat_source["reviewed_snapshot_sha256"]
                or observed.player_identity_status != "unresolved_ign_alias"
                or pd.notna(observed.player_id)
                or pd.isna(roster.player_id)
                or bool(observed.feature_eligible_for_training)
                or bool(observed.feature_eligible_for_s18)
            ):
                raise ValueError("Scope snapshot/status/kelayakan fitur alias tidak cocok.")
            collision = result.loc[
                result.season.eq(entry["season"])
                & result.team_code.eq(entry["team_code"])
                & result.player_id.eq(roster.player_id)
            ]
            if not collision.empty:
                raise ValueError(
                    "Target sudah dipakai observasi lain; tidak menjumlahkan statistik."
                )
            source = entry["roster_source"]
            evidence, html = _cached_source(
                root, official, source["url"], source["reviewed_snapshot_sha256"]
            )
            evidence["kind"] = "official_roster_entry_metadata"
            audit["sources"].append(evidence)
            if html is None or roster.source != source["url"]:
                raise ValueError("Snapshot/URL roster tidak cocok dengan review.")
            soup = BeautifulSoup(html, "html.parser")
            if f"Roster for Season {entry['season']}" not in soup.get_text(" ", strip=True):
                raise ValueError("Musim roster tidak ditemukan dalam HTML yang ditinjau.")
            cards = [
                card for card in soup.select(".team-item")
                if card.select_one(".player-name") is not None
                and card.select_one(".player-name").get_text(" ", strip=True) == entry["roster_ign"]
            ]
            if len(cards) != 1:
                raise ValueError("Entri IGN roster hilang atau ganda pada HTML resmi.")
            image = cards[0].select_one(".team-player-img img[src]")
            asset = urlparse(image["src"]) if image else urlparse("")
            if (
                asset.scheme != "https"
                or asset.hostname != "mplph-storage.sgp1.digitaloceanspaces.com"
                or unquote(asset.path) != source["asset_path"]
                or not asset.path.startswith(
                    f"/season{entry['season']}/player/{entry['team_code']}/"
                )
            ):
                raise ValueError("Aset entri roster tidak cocok dengan scope review.")
            if entry["basis"] == "reviewed_official_roster_asset_alias":
                if Path(unquote(asset.path)).stem.casefold() != entry["statistics_ign"].casefold():
                    raise ValueError("Nama aset resmi tidak mendukung alias yang ditinjau.")
            for source in entry["supporting_sources"]:
                evidence, html = _cached_source(
                    root, supporting, source["url"], source["reviewed_snapshot_sha256"]
                )
                audit["sources"].append({**evidence, "kind": source["kind"]})
                if html is None:
                    raise ValueError("Laporan pendukung alias hilang/rusak.")
                article = BeautifulSoup(html, "html.parser")
                text = " ".join(article.stripped_strings)
                text = " ".join(text.split()).casefold()
                published = article.select_one('meta[property="article:published_time"]')
                if (
                    published is None
                    or pd.Timestamp(published.get("content")).date().isoformat()
                    != source["published_on"]
                    or any(
                        " ".join(term.split()).casefold() not in text
                        for term in source["required_terms"]
                    )
                ):
                    raise ValueError("Isi/tanggal laporan tidak cocok dengan bukti review alias.")
            values = {
                "player_id": roster.player_id,
                "matched_roster_player": roster.player,
                "role": roster.role,
                "player_identity_status": STATUS,
                "player_alias_basis": entry["basis"],
                "player_alias_registry_sha256": sha256(content),
                "player_alias_source_urls": " | ".join(s["url"] for s in audit["sources"]),
                "player_alias_reviewed_on": config["reviewed_on"],
                "player_link_scope": report["scope"],
            }
            for column, value in values.items():
                result.loc[selected.index, column] = value
            audit.update(status="verified", reason=None)
        except (ValueError, KeyError, TypeError) as error:
            audit["reason"] = str(error)
    report.update(
        status="pass" if all(a["status"] == "verified" for a in audits) else "requires_review",
        reviewed_on=config["reviewed_on"],
        information_cutoff=cutoff,
        registry_sha256=sha256(content),
        resolved_aliases=sum(a["status"] == "verified" for a in audits),
        unresolved_ign_rows=int(result.player_identity_status.eq("unresolved_ign_alias").sum()),
        aliases=audits,
    )
    return result, report
