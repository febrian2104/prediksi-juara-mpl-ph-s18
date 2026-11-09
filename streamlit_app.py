import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from mpl_ph_predictor.dashboard import main  # noqa: E402

if __name__ == "__main__":
    main()
