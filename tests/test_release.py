import subprocess
from pathlib import Path

import pytest

from mpl_ph_predictor.release import git_visible_files

ROOT = Path(__file__).resolve().parents[1]


def test_release_keeps_private_local_guides_outside_github_package():
    if (ROOT / ".git").exists():
        paths = set(git_visible_files(ROOT))
        assert Path("README.md") in paths
        assert Path("streamlit_app.py") in paths
        assert Path("PROJECT_STATUS.md") not in paths
        assert not any(path.parts[0] == "docs" for path in paths)
    else:
        assert not (ROOT / "docs").exists()
        assert not (ROOT / "PROJECT_STATUS.md").exists()


def test_release_rejects_tracked_symlink_escaping_allowed_workspace(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "link").symlink_to(tmp_path.parent / "outside")
    subprocess.run(["git", "add", "link"], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match="escapes"):
        git_visible_files(tmp_path)
