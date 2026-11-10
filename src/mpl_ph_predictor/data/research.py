"""Log real HTTP outcomes for missing seasons, including broken official archives."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from mpl_ph_predictor.archive import sha256, utc_now

ROOT = Path(__file__).resolve().parents[3]


def collect_historical_source_availability() -> pd.DataFrame:
    early = pd.read_csv(ROOT / "data/raw/early_match_evidence.csv")
    rosters = pd.read_csv(ROOT / "data/raw/historical_roster_observations.csv")
    candidates: dict[str, dict[str, Any]] = {}
    for season in range(1, 6):
        for suffix in ("", "/schedule"):
            candidates[f"https://ph-mpl.com/s{season}{suffix}"] = {
                "season": season,
                "source_type": "official_archive_probe",
            }
    for row in pd.concat([early, rosters]).itertuples(index=False):
        candidates[row.source_url] = {
            "season": int(row.season),
            "source_type": "historical_evidence_reference",
        }
    candidates.update(
        {
            "https://mpl.mobilelegends.com/2018/ph/": {
                "season": 1,
                "source_type": "legacy_official_site_probe",
            },
            (
                "https://met.events/events/"
                "mobile-legends-bang-bang-professional-league-philippines-season-3/"
            ): {
                "season": 3,
                "source_type": "official_organizer_event",
            },
            "https://www.oneesports.ph/mobile-legends/listahan-mpl-ph-champions/": {
                "season": pd.NA,
                "source_type": "secondary_champion_crosscheck",
            },
            "https://ph-mpl.com/news/whos-who-in-season-8-of-mpl-ph": {
                "season": 8,
                "source_type": "official_franchise_history",
            },
            "https://challonge.com/MPLPHS3RegSeason": {
                "season": 3,
                "source_type": "alternative_bracket_probe",
            },
            "https://liquipedia.net/mobilelegends/MPL/Philippines/Season_5/Regular_Season": {
                "season": 5,
                "source_type": "secondary_match_archive_probe",
            },
        }
    )
    from mpl_ph_predictor.data.early_results import reviewed_batches

    for _, _, batch in reviewed_batches(ROOT):
        for url in [batch["source_url"], *batch.get("supporting_source_urls", [])]:
            candidates[url] = {
                "season": batch["season"],
                "source_type": "reviewed_secondary_evidence",
            }
    coverage_config = json.loads((ROOT / "config/early_season_coverage.json").read_text())
    for season, expectation in coverage_config["seasons"].items():
        for url in expectation["source_urls"]:
            candidates[url] = {"season": int(season), "source_type": "early_coverage_reference"}
    snapshots = ROOT / "data/raw/source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    def inspect(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        url, info = item
        record = {**info, "url": url, "retrieved_at": utc_now()}
        try:
            response = requests.get(url, timeout=20)
            content = response.content
            digest = sha256(content)
            path = snapshots / f"{digest}.html"
            try:
                with path.open("xb") as handle:
                    handle.write(content)
            except FileExistsError:
                pass
            record.update(
                {
                    "http_status": response.status_code,
                    "final_url": response.url,
                    "sha256": digest,
                    "snapshot_path": str(path.relative_to(ROOT)),
                    "retrieval_status": (
                        "accessible"
                        if response.ok and "id-mpl.com" not in response.url
                        else "unavailable_or_wrong_region"
                    ),
                }
            )
        except requests.RequestException as error:
            record.update({"retrieval_status": "network_error", "error": type(error).__name__})
        return record

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(inspect, candidates.items()))
    frame = pd.DataFrame(records)
    frame["accessed_at"] = frame["retrieved_at"].str[:10]
    frame.to_csv(ROOT / "data/raw/historical_source_manifest.csv", index=False)
    (ROOT / "reports/historical_source_availability.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str)
    )
    return frame
