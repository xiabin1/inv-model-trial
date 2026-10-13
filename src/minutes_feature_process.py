from __future__ import annotations

import argparse
from pathlib import Path

from .data.features.minutes_feature_manager import MinutesFeatureManager
from .data.storage import Storage


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate minute features as of each day's cutoff")
    parser.add_argument(
        "--date",
        help="Compute a single day, e.g. 2026-09-24 (default: schedule.start_date..end_date)",
    )
    args = parser.parse_args()

    manager = MinutesFeatureManager(ROOT_DIR / "config/feature/minutes_feature_config.yaml")
    data = manager.calculate_minutes_features(start_date=args.date, end_date=args.date)
    storage = Storage(manager.config["storage"], ROOT_DIR)
    print(
        f"Processed {len(data)} minute feature rows for {data['symbol'].nunique()} symbols "
        f"over {data['date'].nunique()} day(s); saved to {storage.describe(manager.output_dir())}"
    )


if __name__ == "__main__":
    main()
