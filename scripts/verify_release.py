"""Local release verification; every output stays within the PH project."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpl_ph_predictor.release import main  # noqa: E402

if __name__ == "__main__":
    main()
