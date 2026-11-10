"""Read-only source comparison; mismatching counters never rewrite source values."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

from mpl_ph_predictor.archive import sha256, utc_now

ROOT = Path(__file__).resolve().parents[3]
METRICS = {
    "Total Kills": "kills",
    "Total Deaths": "deaths",
    "Total Assists": "assists",
    "Total Gold": "gold",
    "Total Damage": "hero_damage",
}


def _numeric_or_none(value: str) -> float | None:
    try:
        number = float(value.replace(",", "").strip())
    except ValueError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def compare_game_counters(
    pane: Tag,
    players: pd.DataFrame,
    code_aliases: dict[str, str],
) -> list[dict]:
    """Compare explicitly labelled counters, not gold/minute-derived durations.

    Gold/damage may differ because source definitions or observation times differ.
    A difference is a review flag, not evidence which of the source values is wrong.
    """
    names = [tag.get_text(" ", strip=True) for tag in pane.select("#summary .team-name h3")]
    if len(names) != 2 or any(name not in code_aliases for name in names):
        raise ValueError("Identitas dua tim pada ringkasan belum terpetakan.")
    codes = [code_aliases[name] for name in names]
    if len(set(codes)) != 2 or set(codes) != set(players["team_code"]):
        raise ValueError("Tim ringkasan dan scoreboard berbeda.")
    checks = []
    for label, metric in METRICS.items():
        nodes = [
            node
            for node in pane.select("#summary .stats-divider")
            if node.get_text(" ", strip=True) == label
        ]
        cells = nodes[0].parent.select(".stats-num") if len(nodes) == 1 else []
        for index, code in enumerate(codes):
            summary = (
                _numeric_or_none(cells[index].get_text(" ", strip=True))
                if (len(cells) == 2)
                else None
            )
            values = players.loc[players["team_code"].eq(code), metric]
            scoreboard = float(values.sum()) if len(values) == 5 and values.notna().all() else None
            if scoreboard is not None and not math.isfinite(scoreboard):
                scoreboard = None
            status = (
                "not_comparable"
                if summary is None or scoreboard is None
                else "match"
                if summary == scoreboard
                else "difference_requires_review"
            )
            checks.append(
                {
                    "team_code": code,
                    "metric": metric,
                    "summary_value": summary,
                    "scoreboard_sum": scoreboard,
                    "status": status,
                    "difference": scoreboard - summary
                    if scoreboard is not None and summary is not None
                    else None,
                }
            )
    return checks


def audit_game_summaries(*, root: Path = ROOT) -> dict:
    """Use verified local source snapshots only; no network, model training or corrections."""
    required = [
        root / f"data/raw/{name}"
        for name in (
            "official_games.csv",
            "official_player_game_stats.csv",
            "game_detail_source_manifest.csv",
        )
    ]
    if not all(path.is_file() for path in required):
        return {"status": "not_available", "reason": "Detail game lokal belum tersedia."}
    games, players, manifest = (pd.read_csv(path) for path in required)
    matches = pd.read_csv(root / "data/clean/matches.csv").set_index("match_id")
    records = manifest.loc[manifest["retrieval_status"].eq("ok")].drop_duplicates(
        "url", keep="last"
    )
    sources = records.set_index("url").to_dict(orient="index")
    documents, game_rows, counter_rows = {}, [], []
    for row in games.to_dict(orient="records"):
        shared = {column: row[column] for column in ("season", "match_id", "game_number", "source")}
        result = {**shared, "summary_checked_counters": 0, "summary_difference_count": 0}
        try:
            url = row["source"]
            record = sources[url]
            if url not in documents:
                documents.clear()  # Keep one parsed HTML document, not the entire archive in RAM.
                content = (root / record["snapshot_path"]).read_bytes()
                if sha256(content) != record["sha256"]:
                    raise ValueError("Checksum snapshot sumber tidak cocok.")
                documents[url] = BeautifulSoup(content, "html.parser")
            pane = documents[url].select_one(f"#game{int(row['game_number'])}")
            if pane is None:
                raise ValueError("Game tidak ditemukan dalam snapshot sumber.")
            match = matches.loc[row["match_id"]]
            aliases = {}
            for side in ("a", "b"):
                code = str(match[f"team_{side}_code"])
                aliases[code] = code
                aliases[str(match.get(f"raw_team_{side}_code", code))] = code
            observed = players.loc[
                players["match_id"].eq(row["match_id"])
                & players["game_number"].eq(row["game_number"])
            ]
            checks = compare_game_counters(pane, observed, aliases)
            counter_rows.extend(
                {
                    **shared,
                    **check,
                    "source_snapshot_path": record["snapshot_path"],
                    "source_retrieved_at": record["retrieved_at"],
                }
                for check in checks
            )
            checked = sum(check["status"] != "not_comparable" for check in checks)
            differences = sum(check["status"] == "difference_requires_review" for check in checks)
            result.update(
                summary_checked_counters=checked,
                summary_difference_count=differences,
                summary_check_status="difference_requires_review"
                if differences
                else "matched_checked_counters"
                if checked == 10
                else "incomplete_summary",
                reason=None,
            )
        except (KeyError, ValueError, OSError) as error:
            result.update(summary_check_status="summary_unavailable", reason=str(error))
        game_rows.append(result)
    audits = pd.DataFrame(
        game_rows,
        columns=[
            "season",
            "match_id",
            "game_number",
            "source",
            "summary_checked_counters",
            "summary_difference_count",
            "summary_check_status",
            "reason",
        ],
    )
    checks = pd.DataFrame(
        counter_rows,
        columns=[
            "season",
            "match_id",
            "game_number",
            "source",
            "team_code",
            "metric",
            "summary_value",
            "scoreboard_sum",
            "status",
            "difference",
            "source_snapshot_path",
            "source_retrieved_at",
        ],
    )
    folder = root / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    audits.to_csv(folder / "game_summary_audit.csv", index=False)
    checks.to_csv(folder / "game_summary_checks.csv", index=False)
    differences = checks.loc[checks["status"].eq("difference_requires_review")]
    summary = {
        "generated_at": utc_now(),
        "games": len(games),
        "comparable_counters": int(checks["status"].ne("not_comparable").sum()),
        "counter_differences": len(differences),
        "games_with_differences": int(audits["summary_difference_count"].gt(0).sum()),
        "games_without_complete_comparison": int(audits["summary_checked_counters"].lt(10).sum()),
        "differences_by_metric": {
            str(k): int(v) for k, v in differences.groupby("metric").size().items()
        },
        "limitations": "Hanya konsistensi angka yang diperiksa. Selisih gold/damage mungkin "
        "berbeda definisi/waktu observasi; tidak menentukan nilai mana yang benar, memverifikasi "
        "winner/durasi, atau memperbaiki skor/penempatan pemain. Detail tetap non-training.",
    }
    (folder / "game_summary_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary
