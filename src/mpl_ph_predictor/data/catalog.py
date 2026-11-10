"""Reference-compatible tables with explicit missing values and provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from mpl_ph_predictor.data.contracts import conform_table, load_dataset_schema, variable_coverage
from mpl_ph_predictor.data.normalization import slug

ROOT = Path(__file__).resolve().parents[3]


def match_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "match_id" not in result:
        result["match_id"] = result.apply(
            lambda row: f"PH-S{int(row.season):02}-{row.stage}-{row.match_date}-"
            f"{row.team_a_code}-{row.team_b_code}",
            axis=1,
        )
        result["match_id_basis"] = "local_key_from_verified_match_facts"
    if "match_date" in result:
        result["date"] = result["match_date"]
    else:
        result["date"] = (
            pd.to_datetime(result["scheduled_at"], utc=True)
            .dt.tz_convert("Asia/Manila")
            .dt.strftime("%Y-%m-%d")
        )
    result["team_a"] = result["team_a_name"]
    result["team_b"] = result["team_b_name"]
    result["winner"] = result.apply(
        lambda row: "draw"
        if row.team_a_score == row.team_b_score
        else "team_a"
        if row.team_a_score > row.team_b_score
        else "team_b",
        axis=1,
    )
    result["source"] = result["source_url"]
    result["round"] = pd.NA
    result.loc[result["stage"].eq("regular_season"), "round"] = "regular_season"
    result.loc[result["stage"].eq("grand_final"), "round"] = "grand_final"
    if "bracket_match_id" in result:
        bracket = json.loads((ROOT / "config/competition.json").read_text())["playoffs"]["matches"]
        rounds = {match["id"]: match["round"] for match in bracket}
        mask = result["season"].eq(18)
        result.loc[mask, "round"] = (
            result.loc[mask, "bracket_match_id"].map(rounds).fillna(result.loc[mask, "round"])
        )
    result["training_eligible"] = result.get("training_eligible", True)
    return conform_table(result, "matches")


def team_table(matches: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side in ("a", "b"):
        part = matches[["season", f"team_{side}_code", f"team_{side}", "source"]].rename(
            columns={f"team_{side}_code": "team_code", f"team_{side}": "team_name"}
        )
        id_column = f"team_{side}_id"
        part["team_id"] = (
            matches[id_column]
            if id_column in matches
            else part.apply(lambda row: f"ph_s{int(row.season):02}_{row.team_code.lower()}", axis=1)
        )
        part["appearance_date"] = matches["date"]
        parts.append(part)
    result = (
        pd.concat(parts)
        .sort_values("appearance_date")
        .groupby(["season", "team_id", "team_code"], as_index=False)
        .agg(
            team_name=("team_name", "last"),
            historical_names=("team_name", lambda values: " | ".join(dict.fromkeys(values))),
            source=("source", lambda values: " | ".join(sorted(set(values)))),
        )
    )
    result["organization_id"] = pd.NA
    result["franchise_slot_id"] = pd.NA
    result["identity_review_required"] = True
    result["source_url"] = result["source"]
    return conform_table(result, "teams")


def player_table(rosters: pd.DataFrame) -> pd.DataFrame:
    result = rosters.copy()
    result["team"] = result["official_team_name"]
    result["player"] = result["ign"]
    result["source"] = result["source_url"]
    result["role"] = result["role"].replace(
        {
            "unknown": pd.NA,
            "midlaner": "mid_lane",
            "explaner": "exp_lane",
            "goldlaner": "gold_lane",
            "jungle": "jungler",
            "roam": "roamer",
        }
    )
    result["status"] = "OBSERVATION_ONLY_EFFECTIVE_DATE_UNVERIFIED"
    result["team_name"] = result["team"]
    result["organization_id"] = pd.NA
    result["franchise_slot_id"] = pd.NA
    result["nickname"] = result["ign"]
    result["player_key_raw"] = result["ign"].map(slug)
    result["player_key"] = result["player_key_raw"]
    result["player_id"] = result["player_key"]
    result["role_raw"] = rosters["role"]
    result["member_type"] = "player"
    result.loc[result["role"].eq("coach"), "member_type"] = "coach"
    result["valid_from"] = result.get("effective_from", pd.NA)
    result["valid_to"] = result.get("effective_to", pd.NA)
    result["temporal_basis"] = "observation_only_effective_date_unverified"
    result["identity_verification"] = "ign_only"
    return conform_table(result, "players")


def championship_table(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    final = matches.loc[matches["stage"].isin(["playoffs", "grand_final"])].tail(1)
    result = teams[["season", "team_name", "source"]].rename(columns={"team_name": "team"}).copy()
    for column in ("final_rank", "champion", "runner_up"):
        result[column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if len(final):
        row = final.iloc[0]
        winner, loser = (
            (row.team_a, row.team_b) if row.winner == "team_a" else (row.team_b, row.team_a)
        )
        result["champion"] = result["team"].eq(winner).astype("Int64")
        result["runner_up"] = result["team"].eq(loser).astype("Int64")
        result.loc[result["team"].eq(winner), "final_rank"] = 1
        result.loc[result["team"].eq(loser), "final_rank"] = 2
        result["source"] = row.source
        if "verification_status" in row:
            result["verification_status"] = row.verification_status
    result["rank_coverage"] = "finalists_only_other_ranks_unverified"
    return conform_table(result, "championships")


def official_player_stats(players: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "data/raw/source_manifest.csv")
    record = manifest.loc[manifest["url"].eq("https://ph-mpl.com/data")].iloc[0]
    path = ROOT / str(record.get("snapshot_path", "missing"))
    if not path.is_file():
        return conform_table(pd.DataFrame(), "player_season_stats")
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    table = next(
        (
            table
            for table in soup.select("table")
            if "Total Kills" in table.get_text() and "Kill Participation" in table.get_text()
        ),
        None,
    )
    if table is None:
        return conform_table(pd.DataFrame(), "player_season_stats")
    # Normalize only presentation differences; spelling/digit aliases require review.
    lookup = players.copy()
    lookup["normalized_key"] = lookup["player_key"].str.replace("_", "", regex=False)
    if lookup.duplicated(["team_code", "normalized_key"]).any():
        raise ValueError("Normalisasi IGN menghasilkan identitas ganda dalam satu tim.")
    names = lookup.set_index(["team_code", "normalized_key"])["player"].to_dict()
    roles = lookup.set_index(["team_code", "normalized_key"])["role"].to_dict()
    ids = lookup.set_index(["team_code", "normalized_key"])["player_id"].to_dict()
    team_names = players.set_index("team_code")["team"].to_dict()
    numeric_columns = [
        "total_kills",
        "avg_kills",
        "total_deaths",
        "avg_deaths",
        "total_assists",
        "avg_assists",
        "avg_kda",
        "kill_participation_pct",
    ]
    rows = []
    for tr in table.select("tbody tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
        if len(cells) != 10:
            raise ValueError("Skema tabel statistik pemain resmi berubah; audit diperlukan.")
        player, code = cells[:2]
        code = "ONIC" if code == "ONIC PH" else code
        key = (code, slug(player).replace("_", ""))
        if code not in team_names:
            raise ValueError(f"Tim statistik pemain belum terpetakan: {code}")
        row = {
            "season": 18,
            "team": team_names[code],
            "team_code": code,
            "player": player,
            "role": roles.get(key),
            "stat_scope": "official_aggregate_stage_unverified",
            "snapshot_label": "post_season_official_page_observation",
            "snapshot_date": record["accessed_at"],
            "coverage_status": "partial_numeric_aggregate_stage_unverified",
            "source": record["url"],
            "source_snapshot_path": str(path.relative_to(ROOT)),
            "feature_eligible_for_s18": False,
            "feature_eligible_for_training": False,
            "player_id": ids.get(key, pd.NA),
            "matched_roster_player": names.get(key, pd.NA),
            "player_identity_status": "normalized_ign_match"
            if key in names
            else "unresolved_ign_alias",
        }
        for column, value in zip(numeric_columns, cells[2:], strict=True):
            row[column] = (
                pd.NA
                if value in ("-", "", "N/A")
                else float(value.replace("%", "").replace(",", ""))
            )
        rows.append(row)
    return conform_table(pd.DataFrame(rows), "player_season_stats")


def export_reference_layout(
    matches: pd.DataFrame, teams: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    early = pd.read_csv(ROOT / "data/raw/early_match_evidence.csv")
    historical_rosters = pd.read_csv(ROOT / "data/raw/historical_roster_observations.csv")
    s18_players = player_table(rosters)
    stats = official_player_stats(s18_players)
    historical_stat_path = ROOT / "data/raw/historical_player_season_stats.csv"
    historical_stats = (
        conform_table(pd.read_csv(historical_stat_path), "player_season_stats")
        if historical_stat_path.exists()
        else conform_table(pd.DataFrame(), "player_season_stats")
    )
    reconciliation_path = ROOT / "reports/season_reconciliation.json"
    reconciliation = (
        {row["season"]: row for row in json.loads(reconciliation_path.read_text())}
        if reconciliation_path.exists()
        else {}
    )
    details = {}
    for table in ("games", "drafts"):
        path = ROOT / f"data/raw/official_{table}.csv"
        details[table] = (
            conform_table(pd.read_csv(path), table)
            if path.exists()
            else conform_table(pd.DataFrame(), table)
        )
        details[table] = details[table].loc[details[table]["match_id"].isin(matches["match_id"])]
    summary_audit_path = ROOT / "reports/game_summary_audit.csv"
    if summary_audit_path.is_file():
        quality_flags = pd.read_csv(summary_audit_path)
        columns = ["summary_check_status", "summary_checked_counters", "summary_difference_count"]
        details["games"] = (
            details["games"]
            .drop(columns=columns, errors="ignore")
            .merge(
                quality_flags[["season", "match_id", "game_number", *columns]],
                on=["season", "match_id", "game_number"],
                how="left",
                validate="one_to_one",
            )
        )
    stats.to_csv(ROOT / "data/raw/official_s18_player_season_stats.csv", index=False)
    all_tables: dict[str, list[pd.DataFrame]] = {
        table: [] for table in load_dataset_schema()["tables"]
    }
    coverage, variables = [], []
    for season in range(1, 19):
        folder = ROOT / "data" / (f"mpl-season{season}" if season < 18 else "season18")
        folder.mkdir(parents=True, exist_ok=True)
        source = matches.loc[matches["season"].eq(season)].copy()
        partial = early.loc[early["season"].eq(season)].copy()
        season_matches = match_table(source if len(source) else partial)
        season_teams = team_table(season_matches)
        season_players = (
            s18_players
            if season == 18
            else player_table(historical_rosters.loc[historical_rosters["season"].eq(season)])
        )
        tables = {
            "matches": season_matches,
            "teams": season_teams,
            "players": season_players,
            "championships": championship_table(season_matches, season_teams),
            "games": details["games"].loc[details["games"]["season"].eq(season)],
            "drafts": details["drafts"].loc[details["drafts"]["season"].eq(season)],
            "player_season_stats": stats
            if season == 18
            else historical_stats.loc[historical_stats["season"].eq(season)],
        }
        for name, frame in tables.items():
            frame.to_csv(folder / f"mpl_ph_season_{season:02}_{name}.csv", index=False)
            variables.extend(variable_coverage(frame, name, season))
            all_tables[name].append(frame)
        if season == 18:
            live_matches = season_matches.copy()
            live_matches["observed_at"] = live_matches["result_observed_at"]
            live_matches["team_a_franchise_slot_id"] = pd.NA
            live_matches["team_b_franchise_slot_id"] = pd.NA
            conform_table(
                live_matches.loc[live_matches["stage"].eq("regular_season")],
                "schedule_results",
                live=True,
            ).to_csv(folder / "schedule_results.csv", index=False)
            playoff = live_matches.loc[live_matches["stage"].eq("playoffs")].copy()
            playoff["routing_basis"] = "verified_ph_s18_rulebook"
            playoff["best_of_basis"] = "verified_ph_s18_rulebook"
            conform_table(playoff, "playoff_schedule_results", live=True).to_csv(
                folder / "playoff_schedule_results.csv", index=False
            )
            conform_table(season_players, "rosters", live=True).to_csv(
                folder / "rosters.csv", index=False
            )
            conform_table(season_teams, "teams", live=True).to_csv(
                folder / "teams.csv", index=False
            )
        row: dict[str, Any] = {
            "season": season,
            "collected_match_rows": len(season_matches),
            "model_eligible_rows": len(source),
            "coverage_status": "partial" if season <= 5 else "available_archive_loaded",
            "regular_wl_reconciliation": reconciliation.get(season, {}).get(
                "status", "not_checked"
            ),
            "archive_wl_scope": reconciliation.get(season, {}).get(
                "archive_wl_scope", "not_checked"
            ),
            "regular_season_rows": int(season_matches["stage"].eq("regular_season").sum()),
            "playoff_rows": int(season_matches["stage"].isin(["playoffs", "grand_final"]).sum()),
            "player_rows": len(season_players),
            "games_rows": len(tables["games"]),
            "drafts_rows": len(tables["drafts"]),
            "player_stat_rows": len(tables["player_season_stats"]),
            "limitation": (
                "Hasil parsial; waktu mulai dan seluruh laga belum terverifikasi. Tidak dipakai "
                "untuk model biner; S1 mempunyai seri BO2."
                if season <= 5
                else "Seluruh kartu arsip tersedia dimuat; kelengkapan independen, "
                "detail game/draft dan roster bertanggal belum lengkap. Statistik S18 adalah "
                "observasi pascamusim; statistik arsip belum terverifikasi scope/penetapan tim. "
                "Observasi statistik ini tidak digunakan sebagai fitur S18."
            ),
        }
        (folder / "coverage.json").write_text(json.dumps(row, indent=2, ensure_ascii=False))
        coverage.append(row)
    pd.DataFrame(variables).to_csv(ROOT / "reports/dataset_variable_coverage.csv", index=False)
    frame = pd.DataFrame(coverage)
    frame.to_csv(ROOT / "reports/season_coverage.csv", index=False)
    (ROOT / "reports/season_coverage.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False)
    )
    for name, parts in all_tables.items():
        all_columns = list(dict.fromkeys(column for part in parts for column in part))
        populated = [part.dropna(axis=1, how="all") for part in parts if not part.empty]
        combined = pd.concat(populated, ignore_index=True) if populated else pd.DataFrame()
        for column in all_columns:
            if column not in combined:
                combined[column] = pd.NA
        combined = conform_table(combined, name)
        for layer in ("interim/normalized", "processed/canonical"):
            folder = ROOT / "data" / layer
            folder.mkdir(parents=True, exist_ok=True)
            target = combined
            if layer == "processed/canonical" and name == "matches":
                target = combined.loc[combined["training_eligible"].eq(True)]
            target.to_parquet(folder / f"{name}.parquet", index=False)
    player_games = ROOT / "data/raw/official_player_game_stats.csv"
    if player_games.exists():
        player_frame = pd.read_csv(player_games)
        for layer in ("interim/normalized", "processed/canonical"):
            player_frame.to_parquet(
                ROOT / "data" / layer / "player_game_stats.parquet", index=False
            )
    return frame


def export_processed_outputs() -> None:
    for source, target in (("features", "features"), ("predictions", "predictions")):
        folder = ROOT / "data/processed" / target
        folder.mkdir(parents=True, exist_ok=True)
        for path in (ROOT / "data" / source).glob("*.csv"):
            pd.read_csv(path).to_parquet(folder / f"{path.stem}.parquet", index=False)
