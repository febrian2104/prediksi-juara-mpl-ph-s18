import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# pytest's configured nested basetemp requires its parent on a fresh checkout.
(ROOT / ".pytest_cache").mkdir(exist_ok=True)
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
