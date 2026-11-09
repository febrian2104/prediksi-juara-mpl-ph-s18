from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

from mpl_ph_predictor.archive import sha256

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
HISTORICAL_SEASONS = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
SEASON_YEARS = {
    6: 2020,
    7: 2021,
    8: 2021,
    9: 2022,
    10: 2022,
    11: 2023,
    12: 2023,
    13: 2024,
    14: 2024,
    15: 2025,
    16: 2025,
    17: 2026,
    18: 2026,
}
TEAM_PAGES = {
    "APBR": ("AP BREN", "https://ph-mpl.com/team/apbr"),
    "RORA": ("Aurora Gaming PH", "https://ph-mpl.com/team/rora"),
    "ONIC": ("ONIC Philippines", "https://ph-mpl.com/team/onic"),
    "OMG": ("Smart Omega", "https://ph-mpl.com/team/omg"),
    "FLCN": ("Team Falcons PH", "https://ph-mpl.com/team/flcn"),
    "TLPH": ("Team Liquid PH", "https://ph-mpl.com/team/tlph"),
    "TNC": ("TNC Pro Team", "https://ph-mpl.com/team/tnc"),
    "TWIS": ("Twisted Minds PH", "https://ph-mpl.com/team/twis"),
}
PLAYOFF_IDS_BY_SCHEDULE_ORDER = ("M1", "M2", "M3", "M4", "M6", "M5", "M7", "M8")


@dataclass
class CollectionResult:
    matches: pd.DataFrame
    rosters: pd.DataFrame
    sources: pd.DataFrame
    issues: list[dict[str, Any]]
    documents: dict[str, bytes] = field(default_factory=dict)


def _config() -> dict[str, Any]:
    return json.loads((CONFIG_DIR / "competition.json").read_text(encoding="utf-8"))


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "mpl-ph-prediction-research/0.1 (data audit; contact: local project)"}
    )
    return session


def _get_soup(
    session: requests.Session, url: str, documents: dict[str, bytes] | None = None
) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    if documents is not None:
        documents[url] = response.content
    return BeautifulSoup(response.text, "html.parser")


def _clean(node: Tag | None) -> str:
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _score(text: str) -> tuple[int, int] | None:
    match = re.search(r"(?<![-\d])(\d+)\s*:\s*(\d+)(?!\d)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _historical_match_datetime(label: str, season: int) -> pd.Timestamp:
    match = re.search(r"(\d{1,2}\s+[A-Za-z]{3})\s*·\s*(\d{1,2}:\d{2}\s*[AP]M)", label)
    if not match:
        raise ValueError(f"Tidak dapat membaca tanggal pertandingan: {label!r}")
    parsed = datetime.strptime(
        f"{match.group(1)} {SEASON_YEARS[season]} {match.group(2)}", "%d %b %Y %I:%M %p"
    )
    return pd.Timestamp(parsed, tz=ZoneInfo("Asia/Manila"))


def _parse_historical_page(
    soup: BeautifulSoup, season: int, url: str, issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, card in enumerate(soup.select("div.past-matches"), start=1):
        label = _clean(card.select_one(".match-time"))
        teams = card.select(".item-team")
        raw_score = _clean(card.select_one(".item-score"))
        score = _score(raw_score)
        correction = None
        if len(teams) == 2 and (score is None or score[0] == score[1]):
            corrections = json.loads(
                (CONFIG_DIR / "historical_corrections.json").read_text(encoding="utf-8")
            )
            for candidate in corrections["score_corrections"]:
                if (
                    candidate["season"] == season
                    and [_clean(team) for team in teams]
                    == [candidate["team_a_code"], candidate["team_b_code"]]
                    and _historical_match_datetime(label, season)
                    == pd.Timestamp(candidate["scheduled_at"])
                    and raw_score.startswith(candidate["original_score"])
                ):
                    score = (candidate["team_a_score"], candidate["team_b_score"])
                    correction = candidate
                    issues.append(
                        {
                            "severity": "info",
                            "scope": f"season_{season}",
                            "issue": "corrupted_score_corrected_with_documented_evidence",
                            "detail": candidate["verification"],
                            "action": "included_with_explicit_correction_provenance",
                            "source_url": url,
                            "correction_sources": candidate["source_urls"],
                        }
                    )
        if len(teams) != 2 or score is None or score[0] == score[1]:
            issues.append(
                {
                    "severity": "warning",
                    "scope": f"season_{season}",
                    "issue": "historical_match_not_parseable",
                    "detail": (
                        f"Card {index}: {label}; score={_clean(card.select_one('.item-score'))}"
                    ),
                    "action": "excluded_from_canonical_dataset",
                    "source_url": url,
                }
            )
            continue
        team_a, team_b = (_clean(teams[0]), _clean(teams[1]))
        stage = "regular_season" if label.lower().startswith("week") else "playoffs"
        week_match = re.search(r"Week\s+(\d+)", label, flags=re.IGNORECASE)
        scheduled_at = _historical_match_datetime(label, season)
        records.append(
            {
                "season": season,
                "stage": stage,
                "week": int(week_match.group(1)) if week_match else pd.NA,
                "bracket_match_id": pd.NA,
                "scheduled_at": scheduled_at.isoformat(),
                "team_a_code": team_a,
                "team_b_code": team_b,
                "team_a_score": score[0],
                "team_b_score": score[1],
                "best_of": 3 if stage == "regular_season" else (7 if max(score) == 4 else 5),
                "status": "completed",
                "official_match_id": f"S{season:02}-{stage[:2].upper()}-{index:03}",
                "source_url": url,
                "original_score_text": raw_score.split()[0],
                "score_verification": correction["verification_type"] if correction else "official",
                "correction_source_urls": " | ".join(correction["source_urls"])
                if correction
                else "",
            }
        )
    return records


def _parse_schedule(soup: BeautifulSoup, url: str, *, season: int = 18) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    playoff_index = 0
    for index, item in enumerate(soup.select("#schedule-tab-container .schedule-item"), start=1):
        day_group = item.find_parent("div", class_=lambda value: value and "mb-4" in value)
        if not isinstance(day_group, Tag):
            raise ValueError("Grup tanggal jadwal S18 tidak ditemukan.")
        date_text = _clean(day_group.select_one(".match-category-day"))
        time_node = item.select_one(".text-center.mb-0")
        teams = [_clean(node) for node in item.select(".team-name")]
        score = _score(_clean(item.select_one(".text-center.mt-n2")))
        if len(teams) != 2 or score is None:
            raise ValueError(f"Baris jadwal S18 tidak valid pada urutan {index}.")
        timestamp = pd.Timestamp(
            datetime.strptime(f"{date_text} {_clean(time_node)}", "%A, %d %B %Y %I:%M %p"),
            tz=ZoneInfo("Asia/Manila"),
        )
        tab = item.find_parent("div", class_=lambda value: value and "tab-pane" in value)
        tab_id = str(tab.get("id", "")) if isinstance(tab, Tag) else ""
        stage = "playoffs" if tab_id == "playoffs" else "regular_season"
        week_match = re.search(r"week-(\d+)", tab_id)
        bracket_match_id: str | pd._libs.missing.NAType = pd.NA
        if stage == "playoffs" and season == 18:
            bracket_match_id = PLAYOFF_IDS_BY_SCHEDULE_ORDER[playoff_index]
            playoff_index += 1
        link = item.select_one("a[href*='/data/match/']")
        source_match_url = str(link.get("href")) if isinstance(link, Tag) else url
        slug = urlparse(source_match_url).path.rstrip("/").split("/")[-1]
        records.append(
            {
                "season": season,
                "stage": stage,
                "week": int(week_match.group(1)) if week_match else pd.NA,
                "bracket_match_id": bracket_match_id,
                "scheduled_at": timestamp.isoformat(),
                "team_a_code": teams[0],
                "team_b_code": teams[1],
                "team_a_score": score[0],
                "team_b_score": score[1],
                "best_of": 3 if stage == "regular_season" else (7 if max(score) == 4 else 5),
                "status": "completed",
                "official_match_id": slug or f"S18-{index:03}",
                "source_url": source_match_url,
            }
        )
    if season == 18 and playoff_index != 8:
        raise ValueError(f"Jadwal resmi harus memuat 8 laga playoff; ditemukan {playoff_index}.")
    return records


def _parse_s18_schedule(soup: BeautifulSoup, url: str) -> list[dict[str, Any]]:
    return _parse_schedule(soup, url, season=18)


def _parse_roster(soup: BeautifulSoup, code: str, name: str, url: str) -> list[dict[str, Any]]:
    records = []
    for item in soup.select(".team-item"):
        player = _clean(item.select_one(".player-name"))
        role = _clean(item.select_one(".player-role"))
        if not player or not role:
            continue
        records.append(
            {
                "season": 18,
                "team_code": code,
                "official_team_name": name,
                "ign": player,
                "role": role.lower().replace(" ", "_"),
                "is_coach": role.lower() == "coach",
                "effective_from": pd.NA,
                "effective_to": pd.NA,
                "effective_date_status": "unverified",
                "source_url": url,
            }
        )
    return records


def collect_official_data() -> CollectionResult:
    """Collect reproducible official match and S18 roster records."""
    config = _config()
    observed_at = datetime.now(UTC).isoformat()
    accessed_at = datetime.now(ZoneInfo("Asia/Manila")).date().isoformat()
    documents: dict[str, bytes] = {}
    session = _session()
    issues: list[dict[str, Any]] = [
        {
            "severity": "warning",
            "scope": "season_5",
            "issue": "official_archive_has_no_machine_readable_match_cards",
            "detail": "Halaman arsip resmi S5 tidak memuat kartu hasil yang dapat diaudit.",
            "action": "season_excluded_from_training",
            "source_url": "https://ph-mpl.com/s5",
        },
        {
            "severity": "info",
            "scope": "roster_features",
            "issue": "dated_historical_rosters_unavailable_in_collected_official_archives",
            "detail": "Roster S18 tersedia, tetapi roster bertanggal lintas musim belum lengkap.",
            "action": "roster_feature_excluded_from_initial_model",
            "source_url": "https://ph-mpl.com/teams",
        },
    ]
    sources: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for url in ("https://ph-mpl.com/s5", "https://ph-mpl.com/s5/schedule"):
        _get_soup(session, url, documents)
        sources.append(
            {
                "source_type": "official_archive_probe",
                "season": 5,
                "url": url,
                "accessed_at": accessed_at,
                "records_collected": 0,
            }
        )
    for season in HISTORICAL_SEASONS:
        url = f"https://ph-mpl.com/s{season}" + ("/schedule" if season == 8 else "")
        soup = _get_soup(session, url, documents)
        season_records = (
            _parse_schedule(soup, url, season=season)
            if season == 8
            else _parse_historical_page(soup, season, url, issues)
        )
        if not season_records:
            raise ValueError(f"Arsip S{season} kosong; pembaruan dibatalkan.")
        matches.extend(season_records)
        sources.append(
            {
                "source_type": "official_historical_archive",
                "season": season,
                "url": url,
                "accessed_at": accessed_at,
                "records_collected": len(season_records),
            }
        )

    schedule_url = config["official_sources"]["schedule_results"]
    schedule_records = _parse_s18_schedule(
        _get_soup(session, schedule_url, documents), schedule_url
    )
    matches.extend(schedule_records)
    sources.append(
        {
            "source_type": "official_schedule_results",
            "season": 18,
            "url": schedule_url,
            "accessed_at": accessed_at,
            "records_collected": len(schedule_records),
        }
    )

    rosters: list[dict[str, Any]] = []
    for code, (name, url) in TEAM_PAGES.items():
        team_records = _parse_roster(_get_soup(session, url, documents), code, name, url)
        rosters.extend(team_records)
        sources.append(
            {
                "source_type": "official_s18_roster",
                "season": 18,
                "url": url,
                "accessed_at": accessed_at,
                "records_collected": len(team_records),
            }
        )

    for source_type, key in (
        ("official_rulebook", "rules"),
        ("official_standings", "standings"),
        ("official_team_list", "teams"),
        ("publisher_champion_release", "champion"),
    ):
        _get_soup(session, config["official_sources"][key], documents)
        sources.append(
            {
                "source_type": source_type,
                "season": 18,
                "url": config["official_sources"][key],
                "accessed_at": accessed_at,
                "records_collected": pd.NA,
            }
        )

    corrections = json.loads((CONFIG_DIR / "historical_corrections.json").read_text())
    evidence_urls = [
        url for correction in corrections["score_corrections"] for url in correction["source_urls"]
    ] + [
        corrections["season_team_aliases"][0]["source_url"],
        corrections["season_team_aliases"][0]["corroborating_url"],
    ]
    for url in dict.fromkeys(evidence_urls):
        _get_soup(session, url, documents)
        sources.append(
            {
                "source_type": "historical_correction_evidence",
                "season": pd.NA,
                "url": url,
                "accessed_at": accessed_at,
                "records_collected": pd.NA,
            }
        )
    match_frame = pd.DataFrame(matches)
    match_frame["result_observed_at"] = observed_at
    cutoff = pd.Timestamp(config["data_cutoff"], tz=config["timezone"]) + pd.Timedelta(days=1)
    match_frame = match_frame.loc[pd.to_datetime(match_frame["scheduled_at"], utc=True).lt(cutoff)]
    roster_frame = pd.DataFrame(rosters).drop_duplicates(["team_code", "ign", "role"])
    roster_frame["observed_at"] = observed_at
    source_frame = pd.DataFrame(sources)
    source_frame["retrieved_at"] = observed_at
    source_frame["sha256"] = source_frame["url"].map(lambda url: sha256(documents[url]))
    source_frame["snapshot_path"] = source_frame["sha256"].map(
        lambda digest: f"data/raw/source_snapshots/{digest}.html"
    )
    return CollectionResult(
        matches=match_frame,
        rosters=roster_frame,
        sources=source_frame,
        issues=issues,
        documents=documents,
    )


def write_raw_data(result: CollectionResult) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = RAW_DIR / "source_snapshots"
    snapshots.mkdir(exist_ok=True)
    for content in result.documents.values():
        path = snapshots / f"{sha256(content)}.html"
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)
    result.matches.to_csv(RAW_DIR / "official_matches.csv", index=False)
    result.rosters.to_csv(RAW_DIR / "official_s18_rosters.csv", index=False)
    result.sources.to_csv(RAW_DIR / "source_manifest.csv", index=False)
    (RAW_DIR / "collection_issues.json").write_text(
        json.dumps(result.issues, indent=2, ensure_ascii=False), encoding="utf-8"
    )
