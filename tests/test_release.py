import subprocess
from pathlib import Path

import pytest

from mpl_ph_predictor.release import git_visible_files

ROOT = Path(__file__).resolve().parents[1]


def test_release_contains_operating_guides_and_status_not_ignored_local_only_files():
    required = {Path("PROJECT_STATUS.md"), Path("docs/LEAGUE_IDENTITY.md")}
    required.update(
        Path("docs") / f"{name}.md"
        for name in (
            "STRUCTURE",
            "DATA_SOURCES",
            "DATA_DICTIONARY",
            "OPERATIONS",
            "LIVE_PREDICTIONS",
            "ADDING_A_SEASON",
        )
    )
    assert all((ROOT / path).is_file() for path in required)
    if (ROOT / ".git").exists():
        assert required.issubset(set(git_visible_files(ROOT)))


def test_release_rejects_tracked_symlink_escaping_allowed_workspace(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "link").symlink_to(tmp_path.parent / "outside")
    subprocess.run(["git", "add", "link"], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match="escapes"):
        git_visible_files(tmp_path)
