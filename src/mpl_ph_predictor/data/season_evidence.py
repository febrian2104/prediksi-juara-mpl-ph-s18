"""Historical archive observations and independent standings reconciliation.

Archive observations are not assumed to be complete or valid pre-match features.
Conflicting standings never overwrite verified match results automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from mpl_ph_predictor.archive import sha256, utc_now
from mpl_ph_predictor.data.contracts import conform_table
from mpl_ph_predictor.data.league_history import load_league_history

ROOT = Path(__file__).resolve().parents[3]
RECONCILED_STATUSES = {
    "regular_wl_totals_reconciled",
    "season_wl_totals_reconciled",
    "mixed_scope_wl_totals_reconciled",
}
NUMERIC_STATS = [
    "total_kills",
    "avg_kills",
    "total_deaths",
    "avg_deaths",
    "total_assists",
    "avg_assists",
    "avg_kda",
    "kill_participation_pct",
]
STANDING_COLUMNS = [
    "season",
    "table_number",
    "rank",
    "team_code",
    "match_wins",
    "match_losses",
    "game_wins",
    "game_losses",
    "points",
    "source",
    "observed_at",
    "source_snapshot_path",
]


def normalize_archive_code(code: str, season: int) -> str:
    if code == "ONIC PH":
        return "ONIC"
    return "ECHO" if code == "TLPH" and 8 <= season <= 12 else code


def _numeric(value: str) -> float | None:
    if value in ("", "-", "N/A"):
        return None
    number = float(value.replace(",", "").replace("%", ""))
    if not math.isfinite(number) or number < 0:
        raise ValueError("Statistik nonfinite/negatif tidak diterima.")
    return number


def parse_archive_statistics(content: bytes, season: int, provenance: dict) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser")
    tables = [
        t
        for t in soup.select("table")
        if {"PLAYER", "TEAM"}.issubset(
            {th.get_text(" ", strip=True).upper() for th in t.select("thead th")}
        )
    ]
    if not tables:
        return conform_table(pd.DataFrame(), "player_season_stats")
    if len(tables) != 1:
        raise ValueError("Lebih dari satu tabel statistik; scope harus diperiksa.")
    expected = [
        "Player",
        "Team",
        "Total Kills",
        "AVG Kills",
        "Total Deaths",
        "AVG Deaths",
        "Total Assists",
        "AVG Assists",
        "KDA Ratio",
        "Kill Participation",
    ]
    headers = [th.get_text(" ", strip=True).upper() for th in tables[0].select("thead th")]
    average_headers = [
        "PLAYER",
        "TEAM",
        "AVERAGE",
        "TOTAL",
        "KILLS",
        "DEATHS",
        "ASSISTS",
        "KDA",
        "DAMAGE/GOLD",
        "TEAMFIGHT",
        "HERO USED",
    ]
    averages_only = headers == average_headers
    if headers != [value.upper() for value in expected] and not averages_only:
        raise ValueError("Kontrak header statistik historis berubah.")
    rows = []
    for tr in tables[0].select("tbody tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
        if len(cells) != (9 if averages_only else 10) or not cells[0] or not cells[1]:
            raise ValueError("Baris statistik historis tidak lengkap.")
        extras = {}
        quality_issues = []
        if averages_only:
            numbers = dict.fromkeys(NUMERIC_STATS)
            columns = ["avg_kills", "avg_deaths", "avg_assists", "avg_kda"]
            numbers.update(zip(columns, map(_numeric, cells[2:6]), strict=True))
            extras = dict(
                zip(
                    ["damage_to_gold_ratio", "teamfight_pct", "heroes_used_count"],
                    map(_numeric, cells[6:]),
                    strict=True,
                )
            )
            fight, heroes = extras["teamfight_pct"], extras["heroes_used_count"]
            if fight is not None and fight > 100:
                # Quarantine this optional metric, not unrelated valid observations.
                extras["teamfight_pct"] = None
                extras["teamfight_pct_source_value"] = cells[7]
                quality_issues.append("teamfight_pct_out_of_range")
            if heroes is not None and not heroes.is_integer():
                raise ValueError("Jumlah hero bukan integer.")
        else:
            numbers = dict(zip(NUMERIC_STATS, map(_numeric, cells[2:]), strict=True))
        kp = numbers["kill_participation_pct"]
        if kp is not None and kp > 100:
            raise ValueError("Kill participation di luar 0–100%.")
        for column in ("total_kills", "total_deaths", "total_assists"):
            value = numbers[column]
            if value is not None and not value.is_integer():
                raise ValueError("Jumlah kills/deaths/assists bukan integer.")
        code = normalize_archive_code(cells[1], season)
        rows.append(
            {
                "season": season,
                "team": code,
                "team_code": code,
                "raw_team_code": cells[1],
                "player": cells[0],
                "role": None,
                "player_id": None,
                "stat_scope": "official_archive_aggregate_stage_unverified",
                "snapshot_label": "post_season_archive_page_observation",
                "snapshot_date": provenance["retrieved_at"][:10],
                "observed_at": provenance["retrieved_at"],
                "information_cutoff": provenance["information_cutoff"],
                "coverage_status": "archive_observation_team_assignment_and_stage_unverified",
                "team_name_status": "archive_code_not_verified_full_name",
                "team_assignment_status": "archive_label_not_effective_dated_roster",
                "identity_verification": "observed_ign_only",
                "feature_eligible_for_training": False,
                "feature_eligible_for_s18": False,
                "source": provenance["url"],
                "source_snapshot_path": provenance["snapshot_path"],
                "source_schema": "average_statistics_with_grouped_headers"
                if averages_only
                else "totals_and_averages_flat_headers",
                "row_quality_issues": " | ".join(quality_issues) if quality_issues else None,
                **numbers,
                **extras,
            }
        )
    result = conform_table(pd.DataFrame(rows), "player_season_stats")
    if result.duplicated(["season", "team_code", "player"]).any():
        raise ValueError("Observasi IGN ganda dalam tim/musim.")
    return result


def parse_archive_standings(content: bytes, season: int, provenance: dict) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser")
    rows = []
    for index, table in enumerate(soup.select("table"), start=1):
        headers = [th.get_text(" ", strip=True).upper() for th in table.select("thead th")]
        if not {"TEAM", "MATCH W-L", "GAMES W-L"}.issubset(headers):
            continue
        for tr in table.select("tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
            if len(cells) != len(headers):
                raise ValueError("Lebar baris klasemen berubah.")
            values = dict(zip(headers, cells, strict=True))
            pairs = [
                re.fullmatch(r"(\d+)\s*-\s*(\d+)", values[column])
                for column in ("MATCH W-L", "GAMES W-L")
            ]
            if any(pair is None for pair in pairs):
                raise ValueError("Skor W-L klasemen tidak dapat dibaca.")
            match, game = pairs
            rows.append(
                {
                    "season": season,
                    "table_number": index,
                    "rank": values.get("RANK"),
                    "team_code": normalize_archive_code(values["TEAM"], season),
                    "match_wins": int(match[1]),
                    "match_losses": int(match[2]),
                    "game_wins": int(game[1]),
                    "game_losses": int(game[2]),
                    "points": values.get("POINTS", values.get("MATCH POINT")),
                    "source": provenance["url"],
                    "observed_at": provenance["retrieved_at"],
                    "source_snapshot_path": provenance["snapshot_path"],
                }
            )
    return pd.DataFrame(rows, columns=STANDING_COLUMNS)


def _wl_totals(matches: pd.DataFrame) -> dict[str, list[int]]:
    """Outcome counters only; no season-specific points or ranking assumptions."""
    computed = {}
    for match in matches.itertuples():
        for side, other in (("a", "b"), ("b", "a")):
            code = getattr(match, f"team_{side}_code")
            own, opponent = (
                getattr(match, f"team_{side}_score"),
                getattr(match, f"team_{other}_score"),
            )
            if code not in computed:
                computed[code] = [0, 0, 0, 0]
            computed[code] = [
                a + b
                for a, b in zip(
                    computed[code],
                    [int(own > opponent), int(own < opponent), int(own), int(opponent)],
                    strict=True,
                )
            ]
    return computed


def _wl_differences(computed: dict, compared: pd.DataFrame) -> list[dict]:
    fields = ["match_wins", "match_losses", "game_wins", "game_losses"]
    differences = []
    for code in sorted(set(compared.index) | set(computed)):
        if code not in computed or code not in compared.index:
            differences.append(
                {
                    "team_code": code,
                    "field": "team_presence",
                    "canonical": code in computed,
                    "archive": code in compared.index,
                }
            )
            continue
        for index, field in enumerate(fields):
            if computed[code][index] != int(compared.loc[code, field]):
                differences.append(
                    {
                        "team_code": code,
                        "field": field,
                        "canonical": computed[code][index],
                        "archive": int(compared.loc[code, field]),
                    }
                )
    return differences


def reconcile_standings(matches: pd.DataFrame, observed: pd.DataFrame, season: int) -> dict:
    """Separate regular, season-wide and mixed counter scopes without changing results."""
    season_matches = matches.loc[matches["season"].eq(season)]
    regular = season_matches.loc[season_matches["stage"].eq("regular_season")]
    computed = _wl_totals(regular)
    overall = _wl_totals(season_matches)
    fields = ["match_wins", "match_losses", "game_wins", "game_losses"]
    compared = observed.drop_duplicates(["team_code", *fields])
    conflicts = compared["team_code"].duplicated().any()
    compared = compared.drop_duplicates("team_code").set_index("team_code")
    differences = _wl_differences(computed, compared)
    overall_differences = _wl_differences(overall, compared)
    mixed = {
        code: values[:2] + overall[code][2:] for code, values in computed.items() if code in overall
    }
    mixed_differences = _wl_differences(mixed, compared)
    candidates = {
        "regular_season": differences,
        "all_collected_matches": overall_differences,
        "regular_match_wl_and_all_collected_game_wl": mixed_differences,
    }
    eligible = bool(computed) and bool(overall) and not conflicts and not observed.empty
    exact_scopes = [scope for scope, diffs in candidates.items() if eligible and not diffs]
    closest = min(candidates, key=lambda scope: len(candidates[scope]))
    status = (
        "no_standings_available"
        if observed.empty
        else "canonical_regular_matches_missing"
        if regular.empty
        else "regular_wl_totals_reconciled"
        if "regular_season" in exact_scopes
        else "season_wl_totals_reconciled"
        if "all_collected_matches" in exact_scopes
        else "mixed_scope_wl_totals_reconciled"
        if exact_scopes
        else "source_conflict_or_scope_mismatch"
    )
    return {
        "season": season,
        "status": status,
        "canonical_regular_rows": len(regular),
        "standings_team_rows": len(compared),
        "difference_count": len(differences),
        "match_wl_totals_reconciled": bool(computed)
        and not conflicts
        and not any(
            item["field"] in ("team_presence", "match_wins", "match_losses") for item in differences
        ),
        "game_wl_totals_reconciled": bool(computed)
        and not conflicts
        and not any(
            item["field"] in ("team_presence", "game_wins", "game_losses") for item in differences
        ),
        "differences": differences,
        "canonical_all_stage_rows": len(season_matches),
        "exact_matching_scopes": exact_scopes,
        "archive_wl_scope": exact_scopes[0] if exact_scopes else "unresolved",
        "scope_comparison": {
            scope: {"difference_count": len(diffs), "differences": diffs}
            for scope, diffs in candidates.items()
        },
        "closest_counter_scope": closest if eligible else None,
        "residual_difference_count": len(candidates[closest]) if eligible else None,
        "scope_basis": "counter_agreement_with_collected_matches_not_official_scope_declaration",
        "conflicting_duplicate_team_rows": bool(conflicts),
        "scope": "counter_scope_reconciliation_not_individual_match_completeness",
        "note": "Jadwal/playoff belum terbukti lengkap; konflik tidak memperbaiki skor otomatis.",
    }


def collect_season_evidence(
    *,
    root: Path = ROOT,
    cache_only: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    if cache_only and refresh:
        raise ValueError("Mode cache-only tidak dapat digabung dengan refresh.")
    config = json.loads((root / "config/competition.json").read_text())
    urls = {f"https://ph-mpl.com/s{season}/data": season for season in range(1, 18)}
    urls["https://ph-mpl.com/data"] = 18
    history = json.loads((root / "config/team_name_history.json").read_text())
    for transition in history["transitions"]:
        for url in transition["source_urls"]:
            urls[url] = transition["season"]
    league = load_league_history(root)
    for source in league["verification_sources"]:
        urls[source["url"]] = source["season"]
    for event in league["events"]:
        for url in event["source_urls"]:
            urls[url] = event["season"]
    finals_path = root / "config/reviewed_grand_finals.json"
    if finals_path.is_file():
        for final in json.loads(finals_path.read_text())["finals"]:
            for source in final["sources"]:
                urls[source["url"]] = final["season"]
    assignments_path = root / "config/reviewed_scoreboard_assignments.json"
    if assignments_path.is_file():
        for series in json.loads(assignments_path.read_text())["series"]:
            for source in series["sources"]:
                urls[source["url"]] = series["season"]
    aliases_path = root / "config/reviewed_player_aliases.json"
    if aliases_path.is_file():
        for alias in json.loads(aliases_path.read_text())["aliases"]:
            for source in alias["supporting_sources"]:
                urls[source["url"]] = alias["season"]
    game_results_path = root / "config/reviewed_game_results.json"
    if game_results_path.is_file():
        for series in json.loads(game_results_path.read_text())["series"]:
            for source in series["sources"]:
                urls[source["url"]] = series["season"]
    manifest_path = root / "data/raw/season_evidence_source_manifest.csv"
    previous = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    cache, last_attempts = {}, {}
    if not previous.empty:
        for row in previous.to_dict(orient="records"):
            last_attempts[row["url"]] = row
            if row["retrieval_status"] == "ok":
                cache[row["url"]] = row

    def retrieve(item: tuple[str, int]) -> tuple[dict, bytes | None, bool]:
        url, season = item
        if url in cache and not refresh:
            record = cache[url]
            path = root / record["snapshot_path"]
            if path.is_file() and sha256(path.read_bytes()) == record["sha256"]:
                return record, path.read_bytes(), False
            if cache_only:
                return (
                    {"url": url, "season": season, "retrieval_status": "invalid_cached_snapshot"},
                    None,
                    False,
                )
        if cache_only:
            if url in last_attempts:
                return last_attempts[url], None, False
            return {"url": url, "season": season, "retrieval_status": "not_cached"}, None, False
        record = {
            "url": url,
            "season": season,
            "retrieved_at": utc_now(),
            "information_cutoff": config["data_cutoff"],
        }
        try:
            response = requests.get(url, timeout=35)
            record.update(http_status=response.status_code, final_url=response.url)
            response.raise_for_status()
            if urlparse(response.url).hostname != urlparse(url).hostname:
                raise ValueError("Redirect lintas domain; bukti perlu diperiksa.")
            content = response.content
            digest = sha256(content)
            path = root / f"data/raw/source_snapshots/{digest}.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as handle:
                    handle.write(content)
            except FileExistsError:
                if sha256(path.read_bytes()) != digest:
                    raise ValueError("Isi snapshot lokal tidak sesuai hash.") from None
            record.update(
                sha256=digest, snapshot_path=str(path.relative_to(root)), retrieval_status="ok"
            )
            return record, content, True
        except (requests.RequestException, ValueError) as error:
            record.update(retrieval_status="failed", error=str(error))
            return record, None, True

    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(retrieve, urls.items()))
    attempts = [record for record, _, fresh in fetched if fresh]
    if attempts:
        pd.concat([previous, pd.DataFrame(attempts)], ignore_index=True).to_csv(
            manifest_path,
            index=False,
        )
    statistics, standings, audits = [], [], []
    published_statistics, published_standings = set(), set()
    matches = pd.read_csv(root / "data/clean/matches.csv")
    for record, content, _ in fetched:
        if not record["url"].endswith("/data"):
            continue
        season = int(record["season"])
        row = {
            "season": season,
            "source": record["url"],
            "player_stat_rows": 0,
        }
        for key in ("http_status", "retrieved_at", "snapshot_path"):
            value = record.get(key)
            row[f"source_{key}"] = value if value is not None and pd.notna(value) else None
        if content is None:
            row.update(status=record["retrieval_status"], reason="Sumber belum tersedia.")
        else:
            try:
                observed = parse_archive_standings(content, season, record)
                standings.append(observed)
                if len(observed):
                    published_standings.add(season)
                row.update(reconcile_standings(matches, observed, season))
            except ValueError as error:
                row.update(status="standings_parse_failed", reason=str(error))
            if season < 18:  # Preserve the separately audited S18 roster-linked export.
                try:
                    frame = parse_archive_statistics(content, season, record)
                    statistics.append(frame)
                    if len(frame):
                        published_statistics.add(season)
                    row["player_stat_rows"] = len(frame)
                    row["invalid_optional_metric_rows"] = (
                        int(frame["row_quality_issues"].notna().sum()) if len(frame) else 0
                    )
                    row["statistics_status"] = (
                        "observed_scope_unverified" if len(frame) else "absent"
                    )
                except ValueError as error:
                    row.update(statistics_status="rejected", statistics_error=str(error))
        audits.append(row)
    stats = [frame for frame in statistics if len(frame)]
    combined = pd.concat(stats, ignore_index=True) if stats else pd.DataFrame()
    stat_path = root / "data/raw/historical_player_season_stats.csv"
    retained_stat_rows = 0
    if stat_path.exists():
        old = pd.read_csv(stat_path)
        retained = old.loc[~old["season"].isin(published_statistics)]
        retained_stat_rows = len(retained)
        if len(retained):
            combined = (
                pd.concat([retained, combined], ignore_index=True) if len(combined) else retained
            )
    conform_table(combined, "player_season_stats").to_csv(
        stat_path,
        index=False,
    )
    standing_frame = (
        pd.concat(standings, ignore_index=True)
        if standings
        else pd.DataFrame(columns=STANDING_COLUMNS)
    )
    standing_path = root / "data/raw/archive_standings_observations.csv"
    if standing_path.exists():
        old = pd.read_csv(standing_path)
        retained = old.loc[~old["season"].isin(published_standings)]
        if len(retained):
            standing_frame = (
                pd.concat([retained, standing_frame], ignore_index=True)
                if len(standing_frame)
                else retained
            )
    standing_frame.to_csv(standing_path, index=False)
    (root / "reports/season_reconciliation.json").write_text(
        json.dumps(audits, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in audits]
    ).to_csv(
        root / "reports/season_reconciliation.csv",
        index=False,
    )
    return {
        "generated_at": utc_now(),
        "information_cutoff": config["data_cutoff"],
        "archive_player_stat_rows": len(combined),
        "previous_stat_observations_retained": retained_stat_rows,
        "regular_wl_reconciled_seasons": [
            r["season"] for r in audits if r["status"] == "regular_wl_totals_reconciled"
        ],
        "seasons_requiring_review": [
            r["season"] for r in audits if r["status"] not in RECONCILED_STATUSES
        ],
        "resolved_non_regular_counter_scopes": {
            r["season"]: r["archive_wl_scope"]
            for r in audits
            if r["status"] in RECONCILED_STATUSES - {"regular_wl_totals_reconciled"}
        },
        "all_statistics_excluded_from_training": True,
    }


def main() -> None:
    from mpl_ph_predictor.archive import archive_project_state
    from mpl_ph_predictor.data.catalog import export_reference_layout
    from mpl_ph_predictor.data.normalization import build_canonical_data, write_canonical_data

    parser = argparse.ArgumentParser(description="Ambil observasi arsip dan rekonsiliasi klasemen.")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    archive_project_state(reason="before_season_evidence_update")
    summary = collect_season_evidence(cache_only=args.cache_only, refresh=args.refresh)
    matches, teams, rosters, quality = build_canonical_data()
    if quality["status"] != "pass":
        raise RuntimeError("Audit canonical gagal; ekspor dibatalkan.")
    write_canonical_data(matches, teams, rosters, quality)
    export_reference_layout(matches, teams, rosters)
    archive_project_state(reason="after_season_evidence_update")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
