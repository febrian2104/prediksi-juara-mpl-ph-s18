"""Run all dashboard pages offline; collection/training must never happen here."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
LOCAL_TEMP = ROOT / ".cache/smoke_tmp"
LOCAL_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(LOCAL_TEMP)
os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")
tempfile.tempdir = str(LOCAL_TEMP)

from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> None:
    with (
        patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")),
        patch(
            "mpl_ph_predictor.pipeline.run_pipeline",
            side_effect=AssertionError("Training forbidden"),
        ),
    ):
        entrypoint = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
        assert not entrypoint.exception, entrypoint.exception
        for name in ("page_regular_season", "page_playoff", "page_results"):
            code = (
                "import sys\n"
                f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
                f"from mpl_ph_predictor.dashboard import {name}\n{name}()\n"
            )
            page = AppTest.from_string(code, default_timeout=30).run()
            assert not page.exception, page.exception
            print(f"PASS {name}")
    print("PASS entrypoint and three offline dashboard pages; no training or collection")


if __name__ == "__main__":
    main()
