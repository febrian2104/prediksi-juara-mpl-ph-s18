"""Import explicitly reviewed early-season facts without inventing exact dates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mpl_ph_predictor.archive import sha256, utc_now

ROOT = Path(__file__).resolve().parents[3]


def import_reviewed_early_results(*, root: Path = ROOT) -> dict:
    evidence_path = root / "data/raw/reviewed_early_results.json"
    content = evidence_path.read_bytes()
    evidence = json.loads(content)
    if evidence["training_eligible"] or not 1 <= int(evidence["season"]) <= 5:
        raise ValueError("Bukti awal parsial tidak boleh menjadi data training.")
    raw_path = root / "data/raw/early_match_evidence.csv"
    previous = pd.read_csv(raw_path)
    rows = []
    for result in evidence["results"]:
        a, b = result["team_a_score"], result["team_b_score"]
        if not all(isinstance(value, int) and value >= 0 for value in (a, b)) or a == b:
            raise ValueError("Skor bukti hasil baru tidak valid.")
        key = (
            f"PH-S{evidence['season']:02}-W{evidence['week']:02}-"
            f"{result['team_a_code']}-{result['team_b_code']}"
        )
        # Retain the first extraction timestamp when rebuilding the same evidence.
        old = (
            previous.loc[previous.get("match_id", pd.Series(dtype=str)).eq(key)]
            if ("match_id" in previous)
            else pd.DataFrame()
        )
        rows.append(
            {
                **result,
                "match_id": key,
                "season": evidence["season"],
                "stage": evidence["stage"],
                "week": evidence["week"],
                "match_date": None,
                "scheduled_at": None,
                "best_of": None,
                "source_url": evidence["source_url"],
                "verification_status": "reviewed_contemporary_secondary_report",
                "training_eligible": False,
                "exclusion_reason": "missing_exact_match_date_start_time_and_incomplete_season",
                "accessed_at": evidence["reviewed_on"],
                "observed_at": old.iloc[0]["observed_at"] if len(old) else utc_now(),
                "date_range_start": evidence["date_range_start"],
                "date_range_end": evidence["date_range_end"],
                "source_access_method": evidence["source_access_method"],
                "evidence_record_path": str(evidence_path.relative_to(root)),
                "evidence_record_sha256": sha256(content),
                "match_id_basis": "local_reviewed_season_week_pair_key_not_exact_match_date",
            }
        )
    additions = pd.DataFrame(rows)
    if additions.match_id.duplicated().any():
        raise ValueError("Identitas bukti hasil baru ganda.")
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
    combined.to_csv(raw_path, index=False)
    report = {
        "generated_at": utc_now(),
        "reviewed_result_rows": len(additions),
        "total_early_result_rows": len(combined),
        "source_url": evidence["source_url"],
        "evidence_record_sha256": sha256(content),
        "limitation": "Sumber sekunder kontemporer melalui web reader; HTTP langsung 403. "
        "Hanya rentang tanggal pekan diketahui. Jam, tanggal per laga dan BO tidak diisi. "
        "Hasil parsial tidak dimasukkan ke training atau dianggap arsip S5 lengkap.",
    }
    (root / "reports/early_results_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
