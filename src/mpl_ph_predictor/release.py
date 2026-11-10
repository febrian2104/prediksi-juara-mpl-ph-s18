"""Test the Git-visible project in an isolated copy, never the reference workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from mpl_ph_predictor.archive import sha256, utc_now

ROOT = Path(__file__).resolve().parents[2]


def git_visible_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-cz", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = [Path(os.fsdecode(value)) for value in result.stdout.split(b"\0") if value]
    for path in paths:
        resolved = (root / path).resolve()
        if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"Release path escapes the PH project: {path}")
    return sorted({path for path in paths if (root / path).is_file()})


def verify_release(*, root: Path = ROOT, fresh_env: bool = False) -> dict:
    cache = root / ".cache/release_checks"
    cache.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="checkout-", dir=cache))
    paths = git_visible_files(root)
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    temporary = destination / ".cache/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env.update(
        TMPDIR=str(temporary),
        XDG_CACHE_HOME=str(destination / ".cache"),
        PIP_CACHE_DIR=str(root / ".cache/pip"),
        PYTHONNOUSERSITE="1",
        STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
    )
    python = Path(sys.executable)
    commands = []
    if fresh_env:
        # ensurepip's temporary files must also stay in the allowed project tree.
        os.environ["TMPDIR"] = str(temporary)
        tempfile.tempdir = str(temporary)
        venv.EnvBuilder(with_pip=True).create(destination / ".venv")
        python = destination / ".venv/bin/python"
        commands.append([str(python), "-m", "pip", "install", "-r", "requirements-dev.txt"])
    commands.extend(
        [
            [str(python), "-m", "pip", "check"],
            [str(python), "-I", "-m", "pytest", "-q"],
            [str(python), "-I", "scripts/smoke_dashboard.py"],
        ]
    )
    report = {
        "started_at": utc_now(),
        "status": "running",
        "fresh_virtualenv": fresh_env,
        "python_version": sys.version.split()[0],
        "requirements_sha256": sha256((root / "requirements.txt").read_bytes()),
        "copied_files": len(paths),
        "checkout_path": str(destination.relative_to(root)),
        "copy_basis": "git_visible_working_tree_including_uncommitted_changes_not_a_commit",
        "cloud_deployment_performed": False,
        "note": "Pengujian lokal terisolasi, bukan deployment Streamlit Cloud.",
    }
    log: Path | None = None
    try:
        for index, command in enumerate(commands):
            log = destination / f"release-step-{index + 1}.log"
            print(f"Release check {index + 1}/{len(commands)}: {' '.join(command)}", flush=True)
            with log.open("w") as handle:
                subprocess.run(
                    command,
                    cwd=destination,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=900,
                )
            print(log.read_text()[-3000:], flush=True)
        report["status"] = "pass"
    except (subprocess.SubprocessError, OSError) as error:
        report.update(status="fail", error=str(error))
        if log is not None and log.is_file():
            print(log.read_text()[-6000:], flush=True)
        raise
    finally:
        report["finished_at"] = utc_now()
        (root / "reports/release_check.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a Git-visible isolated PH checkout.")
    parser.add_argument("--fresh-env", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_release(fresh_env=args.fresh_env), indent=2))
