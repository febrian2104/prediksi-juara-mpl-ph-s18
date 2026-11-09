"""Prospective forecasts; deliberately offers no backdated live mode."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpl_ph_predictor.live import (  # noqa: E402
    evaluate_live_snapshot,
    project_path,
    save_live_forecast,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simpan atau evaluasi prediksi sebelum laga.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    save = subparsers.add_parser("save")
    save.add_argument("--schedule", required=True, help="CSV jadwal mendatang dalam proyek.")
    save.add_argument("--rules", default="config/competition.json")
    save.add_argument("--period", required=True, help="preseason atau week_01, dan seterusnya.")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    try:
        if args.command == "save":
            path = save_live_forecast(
                project_path(args.schedule), project_path(args.rules), period=args.period
            )
        else:
            path = evaluate_live_snapshot(project_path(args.snapshot))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Dibatalkan: {error}\n")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
