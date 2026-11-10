from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from mpl_ph_predictor.data.identity import apply_team_name_history
from mpl_ph_predictor.data.league_history import annotate_match_participants

ROOT = Path(__file__).resolve().parents[3]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_identity_map() -> pd.DataFrame:
    return pd.read_csv(ROOT / "config" / "team_identity.csv", dtype="string")


def invalid_completed_score_mask(matches: pd.DataFrame) -> pd.Series:
    scores = matches[["team_a_score", "team_b_score"]].apply(pd.to_numeric, errors="coerce")
    needed = pd.to_numeric(matches["best_of"], errors="coerce") // 2 + 1
    return (
        scores.isna().any(axis=1)
        | scores.lt(0).any(axis=1)
        | scores.mod(1).ne(0).any(axis=1)
        | scores.max(axis=1).ne(needed)
        | scores.min(axis=1).ge(needed)
        | ~matches["best_of"].isin([1, 3, 5, 7])
    )


def build_canonical_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Normalize official raw files without inventing missing identities or results."""
    raw_matches = pd.read_csv(ROOT / "data" / "raw" / "official_matches.csv")
    rosters = pd.read_csv(ROOT / "data" / "raw" / "official_s18_rosters.csv")
    identity = load_identity_map()
    lookup_id = identity.set_index("team_code")["canonical_team_id"].to_dict()
    lookup_name = identity.set_index("team_code")["official_name"].to_dict()

    matches = raw_matches.copy()
    matches["scheduled_at"] = pd.to_datetime(matches["scheduled_at"], utc=True)
    aliases = json.loads((ROOT / "config/historical_corrections.json").read_text())
    for side in ("a", "b"):
        code = f"team_{side}_code"
        matches[f"raw_{code}"] = matches[code]
        for alias in aliases["season_team_aliases"]:
            mask = matches["season"].between(alias["first_season"], alias["last_season"]) & matches[
                code
            ].eq(alias["raw_team_code"])
            matches.loc[mask, code] = alias["canonical_code"]
    for side in ("a", "b"):
        code = f"team_{side}_code"
        matches[f"team_{side}_id"] = matches[code].map(lookup_id)
        matches[f"team_{side}_name"] = matches[code].map(lookup_name).fillna(matches[code])
    matches = apply_team_name_history(matches)
    matches = annotate_match_participants(matches)
    matches["winner_side"] = matches.apply(
        lambda row: "team_a" if row.team_a_score > row.team_b_score else "team_b", axis=1
    )
    matches["winner_team_id"] = matches.apply(
        lambda row: row.team_a_id if row.winner_side == "team_a" else row.team_b_id, axis=1
    )
    matches["match_id"] = matches.apply(
        lambda row: f"S{int(row.season):02}-{row.stage[:2].upper()}-{row.official_match_id}", axis=1
    )
    matches = matches.sort_values(["scheduled_at", "match_id"]).reset_index(drop=True)

    unknown_codes = sorted(
        set(matches["team_a_code"]).union(matches["team_b_code"]) - set(identity["team_code"])
    )
    duplicate_match_ids = int(matches["match_id"].duplicated().sum())
    invalid_scores = int(invalid_completed_score_mask(matches).sum())
    missing_team_ids = int(matches[["team_a_id", "team_b_id"]].isna().sum().sum())

    teams = (
        pd.concat(
            [
                matches[["season", "team_a_code", "team_a_id", "team_a_name"]].rename(
                    columns={
                        "team_a_code": "team_code",
                        "team_a_id": "team_id",
                        "team_a_name": "official_name",
                    }
                ),
                matches[["season", "team_b_code", "team_b_id", "team_b_name"]].rename(
                    columns={
                        "team_b_code": "team_code",
                        "team_b_id": "team_id",
                        "team_b_name": "official_name",
                    }
                ),
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["season", "team_code", "team_id"], keep="last")
        .sort_values(["season", "team_code"])
        .reset_index(drop=True)
    )
    rosters["team_id"] = rosters["team_code"].map(lookup_id)
    rosters["player_id"] = rosters["ign"].map(slug)
    rosters["identity_verification"] = "ign_only"

    s18_regular = matches.query("season == 18 and stage == 'regular_season'")
    computed = compute_standings(s18_regular)
    expected = {
        "TLPH": (12, 2, 25, 6),
        "FLCN": (12, 2, 26, 10),
        "ONIC": (9, 5, 21, 14),
        "RORA": (9, 5, 20, 13),
        "OMG": (7, 7, 15, 18),
        "TNC": (4, 10, 14, 22),
        "APBR": (3, 11, 10, 23),
        "TWIS": (0, 14, 3, 28),
    }
    standings_match_official = all(
        tuple(
            computed.set_index("team_code").loc[
                code, ["match_wins", "match_losses", "game_wins", "game_losses"]
            ]
        )
        == values
        for code, values in expected.items()
    )
    quality = {
        "raw_match_rows": int(len(raw_matches)),
        "canonical_match_rows": int(len(matches)),
        "historical_training_rows_before_s18": int((matches["season"] < 18).sum()),
        "s18_regular_rows": int(len(s18_regular)),
        "s18_playoff_rows": int(matches.query("season == 18 and stage == 'playoffs'").shape[0]),
        "s18_roster_rows": int(len(rosters)),
        "s8_match_rows": int(matches["season"].eq(8).sum()),
        "corrected_score_rows": int(
            matches.get("correction_source_urls", pd.Series(dtype=str)).notna().sum()
        ),
        "roster_effective_dates_verified": int(rosters["effective_from"].notna().sum()),
        "duplicate_match_ids": duplicate_match_ids,
        "invalid_scores": invalid_scores,
        "missing_team_ids": missing_team_ids,
        "unknown_team_codes": unknown_codes,
        "s18_standings_match_official": bool(standings_match_official),
        "status": "pass"
        if not unknown_codes
        and duplicate_match_ids == 0
        and invalid_scores == 0
        and missing_team_ids == 0
        and standings_match_official
        else "fail",
    }
    return matches, teams, rosters, quality


def compute_standings(matches: pd.DataFrame) -> pd.DataFrame:
    codes = sorted(set(matches["team_a_code"]).union(matches["team_b_code"]))
    rows = []
    for code in codes:
        as_a = matches[matches["team_a_code"].eq(code)]
        as_b = matches[matches["team_b_code"].eq(code)]
        match_wins = int((as_a["team_a_score"] > as_a["team_b_score"]).sum()) + int(
            (as_b["team_b_score"] > as_b["team_a_score"]).sum()
        )
        played = len(as_a) + len(as_b)
        game_wins = int(as_a["team_a_score"].sum() + as_b["team_b_score"].sum())
        game_losses = int(as_a["team_b_score"].sum() + as_b["team_a_score"].sum())
        rows.append(
            {
                "team_code": code,
                "match_points": match_wins,
                "match_wins": match_wins,
                "match_losses": played - match_wins,
                "game_wins": game_wins,
                "game_losses": game_losses,
                "net_game_wins": game_wins - game_losses,
            }
        )
    result = pd.DataFrame(rows)
    result["h2h_match_wins"] = 0
    result["h2h_net_game_wins"] = 0
    for _, tied_rows in result.groupby(["match_points", "net_game_wins"]):
        tied = set(tied_rows["team_code"])
        if len(tied) < 2:
            continue
        head_to_head = matches.loc[
            matches["team_a_code"].isin(tied) & matches["team_b_code"].isin(tied)
        ]
        for team in tied:
            as_a = head_to_head.loc[head_to_head["team_a_code"].eq(team)]
            as_b = head_to_head.loc[head_to_head["team_b_code"].eq(team)]
            h2h_wins = int((as_a["team_a_score"] > as_a["team_b_score"]).sum()) + int(
                (as_b["team_b_score"] > as_b["team_a_score"]).sum()
            )
            h2h_game_diff = int(
                (as_a["team_a_score"] - as_a["team_b_score"]).sum()
                + (as_b["team_b_score"] - as_b["team_a_score"]).sum()
            )
            result.loc[result["team_code"].eq(team), "h2h_match_wins"] = h2h_wins
            result.loc[result["team_code"].eq(team), "h2h_net_game_wins"] = h2h_game_diff
    result = result.sort_values(
        [
            "match_points",
            "net_game_wins",
            "h2h_match_wins",
            "h2h_net_game_wins",
            "team_code",
        ],
        ascending=[False, False, False, False, True],
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def write_canonical_data(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    rosters: pd.DataFrame,
    quality: dict[str, Any],
) -> None:
    clean_dir = ROOT / "data" / "clean"
    report_dir = ROOT / "reports"
    clean_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(clean_dir / "matches.csv", index=False)
    teams.to_csv(clean_dir / "teams.csv", index=False)
    rosters.to_csv(clean_dir / "rosters.csv", index=False)
    (report_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
