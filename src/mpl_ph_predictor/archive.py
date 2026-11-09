"""Append-only, self-contained snapshots of local prediction artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_snapshot(
    files: dict[str, bytes],
    *,
    kind: str,
    metadata: dict[str, Any],
    directory: Path | None = None,
    key: str | None = None,
) -> Path:
    """Create a bundle exclusively; an existing key is never replaced.

    Hashes detect accidental modification. These local files are not an independent
    timestamp authority or protection against an owner deliberately editing files.
    """
    if not files:
        raise ValueError("Snapshot kosong tidak dapat disimpan.")
    if key is not None and (
        not key or any(c not in "-_0123456789abcdefghijklmnopqrstuvwxyz" for c in key)
    ):
        raise ValueError("Kunci snapshot hanya boleh berisi huruf kecil, angka, - dan _.")
    created_at = utc_now()
    payload = {
        "schema_version": 1,
        "kind": kind,
        "saved_at": created_at,
        "metadata": metadata,
        "files": {
            name: {"sha256": sha256(content), "base64": base64.b64encode(content).decode("ascii")}
            for name, content in sorted(files.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = sha256(encoded)
    envelope = json.dumps(
        {"payload_sha256": digest, "payload": payload}, ensure_ascii=False, indent=2
    ).encode("utf-8")
    target = directory if directory is not None else ROOT / "data/predictions/history"
    target.mkdir(parents=True, exist_ok=True)
    name = key or f"{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}-{digest[:12]}"
    path = target / f"{name}.json"
    with path.open("xb") as handle:
        handle.write(envelope)
    return path


def read_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if sha256(encoded) != envelope["payload_sha256"]:
        raise ValueError(f"Integritas snapshot gagal: {path.name}")
    files = {}
    for name, record in payload["files"].items():
        content = base64.b64decode(record["base64"], validate=True)
        if sha256(content) != record["sha256"]:
            raise ValueError(f"Checksum artefak gagal: {name}")
        files[name] = content
    return payload, files


def archive_project_state(*, reason: str, root: Path = ROOT) -> Path | None:
    paths = [
        *sorted((root / "data/predictions").glob("*.csv")),
        *sorted((root / "reports").glob("*.json")),
        *sorted((root / "config").glob("*.*")),
        *sorted((root / "models").glob("*.joblib")),
        *sorted((root / "artifacts").glob("*.joblib")),
        *sorted((root / "data/raw").glob("*.csv")),
        *sorted((root / "data/raw").glob("*.json")),
        root / "requirements.txt",
    ]
    if not list((root / "data/predictions").glob("s18_*.csv")):
        return None
    files = {str(path.relative_to(root)): path.read_bytes() for path in paths if path.is_file()}
    return save_snapshot(
        files,
        kind="reconstruction_archive",
        metadata={
            "reason": reason,
            "prediction_origin": "reconstructed",
            "note": "Tanggal pengarsipan saat ini; tidak membuktikan prediksi dibuat sebelum laga.",
        },
        directory=root / "data/predictions/history",
    )


def verify_history(root: Path = ROOT) -> dict[str, Any]:
    paths = sorted((root / "data/predictions/history").glob("*.json"))
    paths += sorted((root / "data/predictions/live").glob("*/*.json"))
    paths += sorted((root / "data/predictions/evaluations").glob("*.json"))
    for path in paths:
        read_snapshot(path)
    return {"status": "pass", "snapshots_verified": len(paths)}
