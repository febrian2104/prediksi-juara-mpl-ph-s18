"""Import explicitly reviewed early-season facts without inventing exact dates."""

from __future__ import annotations

import json
from datetime import date
from itertools import combinations
from pathlib import Path

import pandas as pd

from mpl_ph_predictor.archive import sha256, utc_now
from mpl_ph_predictor.data.season_evidence import reconcile_standings
from mpl_ph_predictor.storage import publish_table, publish_text

ROOT = Path(__file__).resolve().parents[3]


def reviewed_batches(root: Path) -> list[tuple[Path, bytes, dict]]:
    """Accept the original single report and explicitly reviewed multi-report files."""
    batches = []
    for path in sorted((root / "data/raw").glob("reviewed_early_results*.json")):
        content = path.read_bytes()
        document = json.loads(content)
        version = document.get("schema_version")
        if version == 1:
            batches.append((path, content, document))
        elif version == 2:
            common = {key: value for key, value in document.items() if key != "batches"}
            batches.extend((path, content, {**common, **batch}) for batch in document["batches"])
        else:
            raise ValueError(f"Versi bukti awal tidak didukung: {path.name}")
    if not batches:
        raise FileNotFoundError("Bukti hasil awal yang telah ditinjau belum tersedia.")
    return batches


def _reviewed_rows(evidence: dict, path: Path, content: bytes, root: Path) -> list[dict]:
    if evidence["training_eligible"] is not False or not 1 <= int(evidence["season"]) <= 5:
        raise ValueError("Bukti awal parsial tidak boleh menjadi data training.")
    if evidence["stage"] not in {"regular_season", "playoffs", "grand_final"}:
        raise ValueError("Tahap bukti pertandingan tidak didukung.")
    verification = evidence.get("verification_status", "reviewed_contemporary_secondary_report")
    if verification not in {"reviewed_contemporary_secondary_report", "reviewed_secondary_archive"}:
        raise ValueError("Jenis verifikasi bukti awal tidak dikenali.")
    start, end = (
        date.fromisoformat(evidence[field]) for field in ("date_range_start", "date_range_end")
    )
    if start > end or end > date.fromisoformat(evidence["reviewed_on"]):
        raise ValueError("Rentang tanggal bukti tidak valid.")
    rows = []
    for result in evidence["results"]:
        a, b = result["team_a_score"], result["team_b_score"]
        if not all(type(value) is int and value >= 0 for value in (a, b)) or a == b:
            raise ValueError("Skor bukti hasil baru tidak valid.")
        if result["team_a_code"] == result["team_b_code"]:
            raise ValueError("Tim tidak boleh melawan dirinya sendiri.")
        week = evidence.get("week")
        if evidence["stage"] == "regular_season":
            # Keep identifiers from the original W2 import stable.
            period = f"W{int(week):02}"
        else:
            period = f"PO-{evidence['round']}"
        key = (
            f"PH-S{evidence['season']:02}-{period}-{result['team_a_code']}-{result['team_b_code']}"
        )
        names = evidence.get("team_names", {})
        row = dict(result)
        for side in ("a", "b"):
            row[f"team_{side}_name"] = (
                result.get(f"team_{side}_name") or names[result[f"team_{side}_code"]]
            )
        row.update(
            match_id=key,
            season=evidence["season"],
            stage=evidence["stage"],
            week=week,
            round=evidence.get("round"),
            # A day-level report may establish a date, never a start/publication time.
            match_date=start.isoformat() if start == end else None,
            scheduled_at=None,
            best_of=None,
            source_url=evidence["source_url"],
            verification_status=verification,
            training_eligible=False,
            exclusion_reason="missing_verified_start_time_and_incomplete_season",
            accessed_at=evidence["reviewed_on"],
            date_range_start=start.isoformat(),
            date_range_end=end.isoformat(),
            source_access_method=evidence["source_access_method"],
            evidence_record_path=str(path.relative_to(root)),
            evidence_record_sha256=sha256(content),
            match_id_basis="local_reviewed_season_week_or_round_pair_key_not_start_time",
        )
        rows.append(row)
    return rows


def audit_early_coverage(matches: pd.DataFrame, *, root: Path = ROOT) -> list[dict]:
    """Count known/missing pairings without deriving any missing match's score."""
    config_path = root / "config/early_season_coverage.json"
    if not config_path.is_file():
        return []
    expectations = json.loads(config_path.read_text())["seasons"]
    standings_path = root / "data/raw/archive_standings_observations.csv"
    standings = pd.read_csv(standings_path) if standings_path.is_file() else pd.DataFrame()
    reports = []
    for season_key, expected in expectations.items():
        season = int(season_key)
        collected = matches.loc[matches.season.eq(season)].copy()
        regular = collected.loc[collected.stage.eq("regular_season")]
        pairs = [tuple(sorted((r.team_a_code, r.team_b_code))) for r in regular.itertuples()]
        possible = set(combinations(sorted(expected["participants"]), 2))
        report = {
            "season": season,
            "regular_rows": len(regular),
            "playoff_rows": int(collected.stage.isin(["playoffs", "grand_final"]).sum()),
            "expected_regular_rows": expected["expected_regular_match_rows"],
            "expectation_status": expected["expectation_status"],
            "expectation_source_urls": expected["source_urls"],
            "missing_regular_pairs": [list(pair) for pair in sorted(possible - set(pairs))],
            "duplicate_regular_pairs": [
                list(pair) for pair in sorted(set(pairs)) if pairs.count(pair) > 1
            ],
            "unexpected_regular_pairs": [list(pair) for pair in sorted(set(pairs) - possible)],
            "training_eligible": False,
            "comparison_dataset": "non_training_reviewed_early_observations_not_canonical_training",
            "status": "partial_reviewed_results_not_training_data",
            "limitation": "Kecocokan penghitung bukan bukti lengkapnya laga, roster, tanggal "
            "efektif atau waktu publikasi. Skor yang hilang tidak dihitung balik dari klasemen.",
        }
        observed = standings.loc[standings.season.eq(season)] if not standings.empty else standings
        if not observed.empty:
            paths = observed.source_snapshot_path.dropna().unique()
            if any(
                not (root / path).is_file() or sha256((root / path).read_bytes()) != Path(path).stem
                for path in paths
            ):
                raise ValueError("Checksum sumber klasemen awal tidak cocok.")
            for side in ("a", "b"):
                col = f"team_{side}_code"
                collected[col] = collected[col].replace(expected.get("counter_aliases", {}))
            report["standings_comparison"] = reconcile_standings(collected, observed, season)
        reports.append(report)
    return reports


def import_reviewed_early_results(*, root: Path = ROOT) -> dict:
    raw_path = root / "data/raw/early_match_evidence.csv"
    previous = pd.read_csv(raw_path)
    rows = []
    sources = []
    for path, content, evidence in reviewed_batches(root):
        rows.extend(_reviewed_rows(evidence, path, content, root))
        sources.append(
            {
                "url": evidence["source_url"],
                "rows": len(evidence["results"]),
                "evidence_record_path": str(path.relative_to(root)),
                "evidence_record_sha256": sha256(content),
            }
        )
    additions = pd.DataFrame(rows)
    if additions.match_id.duplicated().any():
        raise ValueError("Identitas bukti hasil baru ganda.")
    semantic_keys = [
        (
            row["season"],
            row["stage"],
            row.get("week"),
            row.get("round"),
            tuple(sorted((row["team_a_code"], row["team_b_code"]))),
        )
        for row in rows
    ]
    if len(set(semantic_keys)) != len(semantic_keys):
        raise ValueError("Pasangan bukti ganda, termasuk jika urutan tim dibalik.")
    for row in rows:
        old = previous.loc[previous["match_id"].eq(row["match_id"])]
        if len(old):
            fields = (
                "season",
                "stage",
                "team_a_code",
                "team_b_code",
                "team_a_score",
                "team_b_score",
            )
            if len(old) != 1 or any(old.iloc[0][field] != row[field] for field in fields):
                raise ValueError(
                    f"Konflik bukti lama; hasil tidak boleh ditimpa: {row['match_id']}"
                )
        row["observed_at"] = old.iloc[0]["observed_at"] if len(old) else utc_now()
    additions = pd.DataFrame(rows)
    retained = (
        previous.loc[~previous["match_id"].isin(additions.match_id)]
        if ("match_id" in previous)
        else previous
    )
    columns = list(dict.fromkeys([*retained.columns, *additions.columns]))
    combined = pd.concat(
        [retained.dropna(axis=1, how="all"), additions.dropna(axis=1, how="all")],
        ignore_index=True,
    ).reindex(columns=columns)
    coverage = audit_early_coverage(combined, root=root)
    publish_table(combined, raw_path)
    report = {
        "generated_at": utc_now(),
        "reviewed_result_rows": len(additions),
        "total_early_result_rows": len(combined),
        "sources": sources,
        "coverage": coverage,
        "limitation": "Fakta ditinjau dari sumber sekunder kontemporer melalui web reader, "
        "bukan klaim arsip HTML berhasil. Rentang pekan atau tanggal laga dicatat sesuai bukti; "
        "jam mulai dan BO yang tidak terverifikasi tetap kosong. "
        "Hasil parsial tidak dimasukkan ke training atau dianggap arsip S5 lengkap.",
    }
    publish_text(
        json.dumps(report, indent=2, ensure_ascii=False), root / "reports/early_results_audit.json"
    )
    return report
