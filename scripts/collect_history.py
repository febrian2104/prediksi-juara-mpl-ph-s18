"""Audit availability of early MPL PH sources without fabricating missing matches."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpl_ph_predictor.data.research import collect_historical_source_availability  # noqa: E402

if __name__ == "__main__":
    frame = collect_historical_source_availability()
    print(frame[["season", "url", "http_status", "retrieval_status"]].to_string(index=False))
