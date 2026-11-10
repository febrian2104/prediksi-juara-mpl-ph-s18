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


def test_expected_operation_guides_exist() -> None:
    for name in ("STRUCTURE", "DATA_SOURCES", "DATA_DICTIONARY", "OPERATIONS",
                 "LIVE_PREDICTIONS", "ADDING_A_SEASON"):
        assert (ROOT / "docs" / f"{name}.md").is_file()
