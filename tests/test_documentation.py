import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_documentation_and_screenshots_are_present() -> None:
    markdown = ROOT / "README.md"
    links = re.findall(r"\]\(([^)]+)\)", markdown.read_text())
    for link in links:
        if "://" not in link and not link.startswith("#"):
            target = link.split("#", 1)[0]
            assert (ROOT / target).is_file(), f"Broken README link: {link}"


def test_public_readme_is_self_contained_without_private_local_guides() -> None:
    readme = (ROOT / "README.md").read_text()
    for command in ("make run", "make update", "make rebuild", "make check", "make smoke"):
        assert command in readme
    assert "streamlit_app.py" in readme
    assert "requirements.txt" in readme
    links = re.findall(r"\]\(([^)]+)\)", readme)
    assert not any(link.startswith("docs/") or link == "PROJECT_STATUS.md" for link in links)
