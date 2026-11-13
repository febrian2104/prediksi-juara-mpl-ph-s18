"""Retrospective game facts from pinned reports, never inferred from scoreboard ratios."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from mpl_ph_predictor.archive import sha256, utc_now
from mpl_ph_predictor.data.finals import _cached_source

ROOT = Path(__file__).resolve().parents[3]
BASIS = "reviewed_report_game_result"
PROVENANCE = [
    "game_result_basis", "game_result_registry_sha256", "game_result_source_urls",
    "game_result_reviewed_on", "game_result_observed_at", "winner_team_code",
]


def exact_duration_seconds(phrase: str) -> int:
    """Parse explicit minute+second wording only, not rounded minutes or per-minute ratios."""
    clock = re.fullmatch(r"(\d+):([0-5]\d)-minute(?: Game (?:One|Two)| match)?", phrase)
    prose = re.fullmatch(r"(\d+) minutes and (\d+|two|nine) seconds", phrase)
    if clock:
        minutes, seconds = int(clock[1]), int(clock[2])
    elif prose:
        seconds_token = {"two": "2", "nine": "9"}.get(prose[2], prose[2])
        minutes, seconds = int(prose[1]), int(seconds_token)
    else:
        raise ValueError("Durasi harus eksplisit sampai detik; ungkapan lain memerlukan review.")
    if seconds >= 60 or minutes * 60 + seconds <= 0:
        raise ValueError("Durasi game tidak valid.")
    return minutes * 60 + seconds


def annotate_game_results(
    games: pd.DataFrame, matches: pd.DataFrame, *, root: Path = ROOT
) -> tuple[pd.DataFrame, dict]:
    """Enrich existing scoreboard rows without changing raw observations or ML features.

    Reviewed facts fail closed per series. A full explicit winner sequence must agree
    with the verified series, but series scores are never used to fill a missing winner.
    """
    result = games.copy()
    if result.duplicated(["match_id", "game_number"]).any():
        raise ValueError("Game duplikat; pengayaan hasil dibatalkan.")
    for column in PROVENANCE:
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
    for column in ("winner", "duration_minutes", "coverage_status"):
        dtype = "Float64" if column == "duration_minutes" else "string"
        result[column] = result[column].astype(dtype)
        observed = f"{column}_source_value"
        if observed not in result:
            result[observed] = result[column]
        else:
            result[observed] = result[observed].astype(dtype)
        prior = result.game_result_basis.eq(BASIS).fillna(False)
        result.loc[prior, column] = result.loc[prior, observed]
    prior = result.game_result_basis.eq(BASIS).fillna(False)
    result.loc[prior, PROVENANCE] = pd.NA
    result["winner"] = result.winner.astype("string")
    result["duration_minutes"] = result.duration_minutes.astype("Float64")
    if "duration_seconds" not in result:
        result["duration_seconds"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[prior, "duration_seconds"] = pd.NA
    report = {
        "generated_at": utc_now(), "status": "registry_not_available",
        "feature_eligible": False, "series": [],
        "scope": "reviewed_existing_scoreboards_not_complete_history_or_prematch_features",
    }
    path = root / "config/reviewed_game_results.json"
    if not path.is_file():
        return result, report
    content = path.read_bytes()
    config = json.loads(content)
    cutoff = json.loads((root / "config/competition.json").read_text())["data_cutoff"]
    if (
        config["schema_version"] != 1 or config["feature_eligible"] is not False
        or config["information_cutoff"] != cutoff
    ):
        raise ValueError("Registry hasil game tidak valid atau melampaui cutoff.")
    entries = config["series"]
    if len({entry["match_id"] for entry in entries}) != len(entries):
        raise ValueError("Review hasil game berisi seri ganda.")
    for entry in entries:
        if not re.fullmatch(r"[a-f0-9]{64}", entry["scoreboard_sha256"]):
            raise ValueError("Review hasil game memerlukan checksum scoreboard.")
        if pd.Timestamp(entry["match_date"]) > pd.Timestamp(cutoff) or not entry["sources"]:
            raise ValueError("Review hasil game memerlukan sumber dan tanggal dalam cutoff.")
        for source in entry["sources"]:
            if (
                source["kind"] != "secondary_contemporary_report"
                or not source["required_terms"]
                or not re.fullmatch(r"[a-f0-9]{64}", source["reviewed_snapshot_sha256"])
                or pd.Timestamp(source["published_on"]) > pd.Timestamp(cutoff)
            ):
                raise ValueError("Laporan hasil game harus bertanggal dan ber-checksum.")
    manifests = {}
    for name in ("game_detail_source_manifest", "season_evidence_source_manifest"):
        file = root / f"data/raw/{name}.csv"
        manifests[name] = pd.read_csv(file) if file.is_file() else pd.DataFrame()
    audits = []
    for entry in entries:
        audit = {"match_id": entry["match_id"], "status": "requires_review", "sources": []}
        audits.append(audit)
        try:
            candidates = matches.loc[matches.match_id.eq(entry["match_id"])]
            if len(candidates) != 1:
                raise ValueError("Identitas pertandingan hilang/ganda.")
            match = candidates.iloc[0]
            fields = ("season", "team_a_code", "team_b_code", "team_a_score", "team_b_score")
            date = pd.Timestamp(match.scheduled_at).tz_convert("Asia/Manila").date().isoformat()
            if any(match[field] != entry[field] for field in fields) or date != entry["match_date"]:
                raise ValueError("Fakta pertandingan berbeda dari review hasil game.")
            selected = result.loc[result.match_id.eq(entry["match_id"])].sort_values("game_number")
            numbers = list(range(1, int(match.team_a_score + match.team_b_score) + 1))
            if (
                selected.game_number.tolist() != numbers
                or selected.season.ne(entry["season"]).any()
                or selected.date.ne(date).any()
                or selected.source.ne(match.source_url).any()
            ):
                raise ValueError("Scoreboard lengkap yang sesuai pertandingan belum tersedia.")
            for column in ("feature_eligible_for_s18", "feature_eligible_for_training"):
                if column not in selected or not selected[column].eq(False).all():
                    raise ValueError("Hasil game retrospektif tidak boleh menjadi fitur pra-laga.")
            facts = entry["games"]
            if [fact["game_number"] for fact in facts] != numbers or any(
                type(fact["game_number"]) is not int for fact in facts
            ):
                raise ValueError("Urutan pemenang harus eksplisit untuk setiap game.")
            expected = {
                match.team_a_code: match.team_a_score, match.team_b_code: match.team_b_score,
            }
            wins = Counter(fact["winner_code"] for fact in facts)
            if any(code not in expected for code in wins) or any(
                wins[code] != score for code, score in expected.items()
            ):
                raise ValueError("Pemenang game berbeda dari skor seri yang diverifikasi.")
            needed = int(match.best_of) // 2 + 1
            prefix = Counter(fact["winner_code"] for fact in facts[:-1])
            if max(prefix.values(), default=0) >= needed or max(wins.values()) != needed:
                raise ValueError("Urutan game melewati saat seri seharusnya sudah berakhir.")
            board, html = _cached_source(
                root, manifests["game_detail_source_manifest"], match.source_url,
                entry["scoreboard_sha256"],
            )
            audit["scoreboard"] = board
            if html is None or selected.source_snapshot_path.ne(board["snapshot_path"]).any():
                raise ValueError("Snapshot scoreboard tidak cocok dengan review hasil game.")
            texts = []
            for source in entry["sources"]:
                evidence, html = _cached_source(
                    root, manifests["season_evidence_source_manifest"], source["url"],
                    source["reviewed_snapshot_sha256"],
                )
                audit["sources"].append({**evidence, "kind": source["kind"]})
                if html is None:
                    raise ValueError("Laporan hasil game hilang atau rusak.")
                text = " ".join(BeautifulSoup(html, "html.parser").stripped_strings)
                text = " ".join(text.split()).casefold()
                if any(
                    " ".join(term.split()).casefold() not in text
                    for term in source["required_terms"]
                ):
                    raise ValueError("Isi laporan hasil game berbeda dari review.")
                texts.append(text)
            combined = " ".join(texts)
            updates = []
            for index, fact in zip(selected.index, facts, strict=True):
                if not fact["evidence_terms"] or any(
                    " ".join(term.split()).casefold() not in combined
                    for term in fact["evidence_terms"]
                ):
                    raise ValueError("Bukti pemenang game belum cocok dengan laporan.")
                clock = fact["duration_clock"]
                seconds = None
                if clock is not None:
                    if not isinstance(clock, str) or not re.fullmatch(r"\d+:[0-5]\d", clock):
                        raise ValueError("Jam durasi tidak valid; wajib menit:detik.")
                    minutes, remainder = map(int, clock.split(":"))
                    seconds = exact_duration_seconds(fact["duration_evidence"])
                    if (
                        seconds != minutes * 60 + remainder
                        or fact["duration_evidence"].casefold() not in combined
                    ):
                        raise ValueError("Durasi tidak cocok dengan kutipan waktu yang eksplisit.")
                winner = "team_a" if fact["winner_code"] == match.team_a_code else "team_b"
                if pd.notna(result.loc[index, "winner"]) and result.loc[index, "winner"] != winner:
                    raise ValueError("Hasil sumber asli bertentangan; tidak ditimpa otomatis.")
                old_duration = result.loc[index, "duration_minutes"]
                if pd.notna(old_duration) and (
                    seconds is None or abs(old_duration * 60 - seconds) > 1e-7
                ):
                    raise ValueError("Durasi sumber asli bertentangan; tidak ditimpa otomatis.")
                updates.append((index, winner, fact["winner_code"], seconds))
            for index, winner, code, seconds in updates:
                result.loc[index, "winner"] = winner
                result.loc[index, "winner_team_code"] = code
                if seconds is not None:
                    result.loc[index, "duration_seconds"] = seconds
                    result.loc[index, "duration_minutes"] = seconds / 60
                result.loc[index, "game_result_basis"] = BASIS
                result.loc[index, "game_result_registry_sha256"] = sha256(content)
                result.loc[index, "game_result_source_urls"] = " | ".join(
                    source["url"] for source in entry["sources"]
                )
                result.loc[index, "game_result_reviewed_on"] = config["reviewed_on"]
                result.loc[index, "game_result_observed_at"] = max(
                    source["retrieved_at"] for source in audit["sources"]
                )
                result.loc[index, "coverage_status"] = (
                    "scoreboard_reviewed_winner_and_exact_duration" if seconds is not None
                    else "scoreboard_reviewed_winner_duration_unverified"
                )
            audit.update(status="verified", games=len(updates), reason=None)
        except (ValueError, KeyError, TypeError) as error:
            audit["reason"] = str(error)
    reviewed = result.game_result_basis.eq(BASIS).fillna(False)
    report.update(
        status="pass" if all(row["status"] == "verified" for row in audits) else "requires_review",
        information_cutoff=cutoff, reviewed_on=config["reviewed_on"],
        registry_sha256=sha256(content), series=audits,
        verified_game_winners=int(reviewed.sum()),
        verified_exact_durations=int((reviewed & result.duration_seconds.notna()).sum()),
        limitation="Bukan bukti seluruh durasi, waktu publikasi hasil atau draft lengkap. "
        "Tidak mengubah training, prediksi historis atau tiebreak S18.",
    )
    return result, report
