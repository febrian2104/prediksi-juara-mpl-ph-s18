"""Conservative extraction of official game scoreboards, never inferred draft order."""

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
from mpl_ph_predictor.data.scoreboard_assignments import (
    ASSIGNMENT_COLUMNS,
    apply_reviewed_assignments,
    load_reviewed_assignments,
    verify_review_scope,
)
from mpl_ph_predictor.storage import publish_table, publish_text

ROOT = Path(__file__).resolve().parents[3]
PLAYER_COLUMNS = [
    "season",
    "match_id",
    "game_number",
    "date",
    "team",
    "team_code",
    "player",
    "hero",
    "kills",
    "deaths",
    "assists",
    "official_kda",
    "gold",
    "hero_damage",
    "damage_taken",
    "tower_damage",
    "source",
    "source_snapshot_path",
    "observed_at",
    "information_cutoff",
    "feature_eligible_for_s18",
    "identity_verification",
    "feature_eligible_for_training",
] + ASSIGNMENT_COLUMNS


def _number(value: str) -> int | float | None:
    cleaned = value.strip().replace(",", "")
    if cleaned in ("", "-", "N/A"):
        return None
    number = float(cleaned)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Statistik negatif/nonfinite pada scoreboard resmi.")
    return int(number) if number.is_integer() else number


def parse_game_detail(
    content: bytes,
    match: dict[str, Any],
    provenance: dict[str, Any],
    *,
    review: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Only publish a series when game counts and both five-player tables agree.

    Heroes are observed selections, not a chronological pick/ban transcript. No
    duration is reconstructed from rounded gold/minute or damage/minute values.
    Game winners are not inferred from gold, kills, towers or the series score.
    """
    if review is not None:
        verify_review_scope(content, match, review)
    soup = BeautifulSoup(content, "html.parser")
    panes = soup.find_all("div", id=re.compile(r"^game\d+$"))
    expected = int(match["team_a_score"] + match["team_b_score"])
    if sorted(int(pane["id"][4:]) for pane in panes) != list(range(1, expected + 1)):
        raise ValueError(f"Jumlah game resmi tidak cocok dengan skor seri ({expected}).")
    codes: dict[str, str] = {}
    names = {}
    for side in ("a", "b"):
        code = str(match[f"team_{side}_code"])
        codes[code] = code
        raw = match.get(f"raw_team_{side}_code", code)
        if pd.notna(raw):
            codes[str(raw)] = code
        names[code] = match[f"team_{side}_name"]
    shared = {
        "season": int(match["season"]),
        "match_id": match["match_id"],
        "date": pd.Timestamp(match["scheduled_at"]).tz_convert("Asia/Manila").date().isoformat(),
        "source": match["source_url"],
        "source_snapshot_path": provenance["snapshot_path"],
        "observed_at": provenance["retrieved_at"],
        "information_cutoff": provenance["information_cutoff"],
        "feature_eligible_for_s18": False,
        "feature_eligible_for_training": False,
    }
    games, drafts, players = [], [], []
    for pane in panes:
        number = int(pane["id"][4:])
        game_shared = {**shared, "game_number": number}
        tables = pane.select("table.table-scoreboard")
        if len(tables) != 2:
            raise ValueError(f"Game {number}: harus ada dua scoreboard tim.")
        seen_codes = []
        game_players = []
        for table in tables:
            container = table.find_parent("div", class_="container")
            logo = container.select_one("img[alt$=' logo']") if container else None
            raw_code = str(logo["alt"])[:-5].strip() if logo else ""
            if raw_code not in codes:
                raise ValueError(f"Game {number}: identitas tim belum terpetakan: {raw_code}")
            code = codes[raw_code]
            seen_codes.append(code)
            headers = [th.get_text(" ", strip=True) for th in table.select("thead th")]
            required = [
                "Player",
                "Hero",
                "Kill",
                "Death",
                "Assist",
                "KDA",
                "Gold",
                "Items",
                "Emblems",
                "Talents",
                "Hero Damage",
                "Damage Taken",
                "Tower Damage",
            ]
            if headers != required:
                raise ValueError(f"Game {number}: skema scoreboard berubah.")
            rows = table.select("tbody tr")
            if review is None and len(rows) != 5:
                raise ValueError(f"Game {number}: scoreboard tidak berisi lima pemain.")
            for tr in rows:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
                if len(cells) != len(required) or not cells[0] or not cells[1]:
                    raise ValueError(f"Game {number}: baris pemain tidak lengkap.")
                row = {
                    **game_shared,
                    "team": names[code],
                    "team_code": code,
                    "source_team_code": raw_code,
                    "source_canonical_team_code": code,
                    "team_assignment_status": "official_scoreboard_table_unreviewed_assignment",
                    "player": cells[0],
                    "hero": cells[1],
                    "identity_verification": "observed_ign_only_not_linked_to_roster",
                }
                indices = {
                    "kills": 2,
                    "deaths": 3,
                    "assists": 4,
                    "official_kda": 5,
                    "gold": 6,
                    "hero_damage": 10,
                    "damage_taken": 11,
                    "tower_damage": 12,
                }
                row.update({column: _number(cells[index]) for column, index in indices.items()})
                game_players.append(row)
        if set(seen_codes) != set(names) or len(set(seen_codes)) != 2:
            raise ValueError(f"Game {number}: dua tim berbeda yang dijadwalkan tidak ditemukan.")
        if review is not None:
            apply_reviewed_assignments(game_players, number, names, review)
        counts = pd.Series([row["team_code"] for row in game_players]).value_counts().to_dict()
        if counts != dict.fromkeys(names, 5):
            raise ValueError(f"Game {number}: scoreboard tidak berisi lima pemain per tim.")
        if pd.DataFrame(game_players).duplicated(["team_code", "player"]).any():
            raise ValueError(f"Game {number}: pemain duplikat dalam tim.")
        for row in game_players:
            drafts.append(
                {
                    **game_shared,
                    "team": row["team"],
                    "team_code": row["team_code"],
                    "action": "pick",
                    "order": None,
                    "hero": row["hero"],
                    "scope": "observed_game_lineup_not_draft_sequence",
                    "player": row["player"],
                    **{column: row.get(column) for column in ASSIGNMENT_COLUMNS},
                }
            )
        players.extend(game_players)
        games.append(
            {
                **game_shared,
                "team_a": match["team_a_name"],
                "team_b": match["team_b_name"],
                "winner": None,
                "duration_minutes": None,
                "coverage_status": "scoreboard_available_winner_and_duration_unverified",
                "reviewed_assignment_rows": sum(
                    row["team_assignment_status"] == "reviewed_contemporary_report_assignment"
                    for row in game_players
                ),
            }
        )
    return games, drafts, players


def collect_game_details(
    *,
    root: Path = ROOT,
    seasons: list[int] | None = None,
    refresh: bool = False,
    cache_only: bool = False,
) -> dict[str, Any]:
    """Fetch known official detail links only; append retrieval attempts to manifest."""
    if refresh and cache_only:
        raise ValueError("--refresh dan --cache-only tidak dapat digabung.")
    matches = pd.read_csv(root / "data/clean/matches.csv")
    config = json.loads((root / "config/competition.json").read_text())
    cutoff = config["data_cutoff"]
    reviews = load_reviewed_assignments(root, cutoff)
    date_limit = pd.Timestamp(cutoff, tz=config["timezone"]) + pd.Timedelta(days=1)
    matches = matches.loc[pd.to_datetime(matches["scheduled_at"], utc=True).lt(date_limit)]
    if seasons is not None:
        matches = matches.loc[matches["season"].isin(seasons)]
    # Archive cards without a genuine detail link are deliberately not guessed.
    eligible = matches.loc[
        matches["source_url"].map(
            lambda url: urlparse(url).hostname == "ph-mpl.com"
            and "/data/match/" in urlparse(url).path
        )
    ]
    raw_dir = root / "data/raw"
    manifest_path = raw_dir / "game_detail_source_manifest.csv"
    previous = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    cache = {}
    if not previous.empty:
        for record in previous.to_dict(orient="records"):
            if record["retrieval_status"] == "ok":
                cache[record["url"]] = record

    def retrieve(match: dict[str, Any]) -> tuple[dict, bytes | None, bool]:
        url = match["source_url"]
        if url in cache and not refresh:
            record = cache[url]
            path = root / record["snapshot_path"]
            if path.is_file() and sha256(path.read_bytes()) == record["sha256"]:
                return {**record, "information_cutoff": cutoff}, path.read_bytes(), False
            # Never silently consume a corrupted/missing cached snapshot.
            if cache_only:
                return {"url": url, "retrieval_status": "invalid_cached_snapshot"}, None, False
        if cache_only:
            return {"url": url, "retrieval_status": "not_cached"}, None, False
        record = {
            "url": url,
            "season": match["season"],
            "match_id": match["match_id"],
            "retrieved_at": utc_now(),
            "information_cutoff": cutoff,
        }
        try:
            response = requests.get(url, timeout=45, headers={"User-Agent": "MPL-PH-Research/1.0"})
            record.update(http_status=response.status_code, final_url=response.url)
            response.raise_for_status()
            if urlparse(response.url).hostname != "ph-mpl.com":
                raise ValueError("Halaman detail diarahkan ke domain lain.")
            content = response.content
            digest = sha256(content)
            path = raw_dir / "source_snapshots" / f"{digest}.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as handle:
                    handle.write(content)
            except FileExistsError:
                if sha256(path.read_bytes()) != digest:
                    raise ValueError("Snapshot lokal tidak sesuai dengan hash sumber.") from None
            record.update(
                sha256=digest,
                snapshot_path=str(path.relative_to(root)),
                retrieval_status="ok",
            )
            return record, content, True
        except (requests.RequestException, ValueError) as error:
            record.update(retrieval_status="failed", error=str(error))
            return record, None, True

    records = eligible.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(retrieve, records))
    attempts = [record for record, _, fresh in fetched if fresh]
    if attempts:
        publish_table(
            pd.concat([previous, pd.DataFrame(attempts)], ignore_index=True), manifest_path
        )
    games, drafts, players, audit = [], [], [], []
    for match, (record, content, _) in zip(records, fetched, strict=True):
        row = {
            "season": int(match["season"]),
            "match_id": match["match_id"],
            "source": match["source_url"],
            "expected_games": int(match["team_a_score"] + match["team_b_score"]),
            "parsed_games": 0,
        }
        try:
            if content is None:
                raise ValueError(str(record["retrieval_status"]))
            parsed_games, parsed_drafts, parsed_players = parse_game_detail(
                content, match, record, review=reviews.get(match["match_id"])
            )
            games.extend(parsed_games)
            drafts.extend(parsed_drafts)
            players.extend(parsed_players)
            row.update(
                status="pass",
                parsed_games=len(parsed_games),
                reviewed_assignment_rows=sum(g["reviewed_assignment_rows"] for g in parsed_games),
                reason=None,
            )
        except (ValueError, KeyError, TypeError) as error:
            row.update(status="unavailable_or_failed_audit", reason=str(error))
        audit.append(row)
    attempted_ids = {record["match_id"] for record in records}
    unavailable_ids = {
        match["match_id"]
        for match, (_, content, _) in zip(
            records,
            fetched,
            strict=True,
        )
        if content is None
    }
    retained_games = {}
    game_path = raw_dir / "official_games.csv"
    if game_path.exists():
        old = pd.read_csv(game_path)
        retained_games = (
            old.loc[old["match_id"].isin(unavailable_ids)].groupby("match_id").size().to_dict()
        )
    for row in audit:
        row["retained_previous_games"] = int(retained_games.get(row["match_id"], 0))
    # Retrieval failure is not evidence that a previously observed fact disappeared.
    # A successfully fetched but invalid scoreboard is still rejected, not silently retained.
    for name, rows in (("games", games), ("drafts", drafts), ("player_game_stats", players)):
        path = raw_dir / f"official_{name}.csv"
        frame = (
            pd.DataFrame(rows, columns=PLAYER_COLUMNS)
            if name == "player_game_stats"
            else conform_table(pd.DataFrame(rows), name)
        )
        if path.exists():
            old = pd.read_csv(path)
            retained = old.loc[
                ~old["match_id"].isin(attempted_ids) | old["match_id"].isin(unavailable_ids)
            ]
            if not retained.empty:
                frame = (
                    retained.copy()
                    if frame.empty
                    else pd.concat([retained, frame], ignore_index=True)
                )
        publish_table(frame, path)
    audit_path = root / "reports/game_detail_audit.csv"
    audit_frame = pd.DataFrame(
        audit,
        columns=[
            "season",
            "match_id",
            "source",
            "expected_games",
            "parsed_games",
            "status",
            "reason",
            "retained_previous_games",
            "reviewed_assignment_rows",
        ],
    )
    if audit_path.exists():
        old = pd.read_csv(audit_path)
        retained = old.loc[~old["match_id"].isin(attempted_ids)]
        if not retained.empty:
            audit_frame = pd.concat([retained, audit_frame], ignore_index=True)
    publish_table(audit_frame, audit_path)
    summary = {
        "information_cutoff": cutoff,
        "generated_at": utc_now(),
        "detail_links_attempted": len(audit_frame),
        "series_passed": int(audit_frame["status"].eq("pass").sum()),
        "series_unavailable_or_failed": int(audit_frame["status"].ne("pass").sum()),
        "games": len(pd.read_csv(game_path)),
        "retained_previous_games": int(audit_frame["retained_previous_games"].fillna(0).sum()),
        "reviewed_assignment_rows": int(audit_frame["reviewed_assignment_rows"].fillna(0).sum()),
        "assignment_reviews": [
            {
                "match_id": review["match_id"],
                "reviewed_on": review["reviewed_on"],
                "registry_sha256": review["registry_sha256"],
                "scoreboard_sha256": review["scoreboard_sha256"],
                "evidence_verified": review["verified"],
                "sources": review["evidence"],
                "feature_eligible": False,
            }
            for review in reviews.values()
        ],
        "limitation": "Tidak membuktikan arsip lengkap. Winner, durasi, ban dan urutan pick "
        "belum terverifikasi. IGN bukan identitas pemain lintas musim. Data observasi "
        "pascamusim ini tidak digunakan sebagai fitur S18.",
    }
    publish_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        root / "reports/game_detail_audit.json",
    )
    from mpl_ph_predictor.data.game_audit import audit_game_summaries

    audit_game_summaries(root=root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ambil scoreboard game resmi, tanpa training.")
    parser.add_argument("--seasons", nargs="+", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    from mpl_ph_predictor.archive import archive_project_state
    from mpl_ph_predictor.data.catalog import export_reference_layout
    from mpl_ph_predictor.data.normalization import build_canonical_data

    archive_project_state(reason="before_game_detail_update")
    summary = collect_game_details(
        seasons=args.seasons,
        refresh=args.refresh,
        cache_only=args.cache_only,
    )
    matches, teams, rosters, quality = build_canonical_data()
    if quality["status"] != "pass":
        raise RuntimeError("Audit canonical gagal; ekspor detail dibatalkan.")
    export_reference_layout(matches, teams, rosters)
    archive_project_state(reason="after_game_detail_update")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
