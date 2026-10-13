from __future__ import annotations

import argparse
from pathlib import Path

from .data.features.daily_feature_manager import DailyFeatureManager
from .data.storage import Storage


ROOT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate daily stock features")
    parser.add_argument("--start-date", help="Start date, e.g. 2026-09-20 or 20260920")
    parser.add_argument("--end-date", help="End date, e.g. 2026-09-20 or 20260920")
    args = parser.parse_args()

    config_path = ROOT_DIR / "config/feature/daily_feature_config.yaml"
    manager = DailyFeatureManager(config_path)
    features = manager.calculate_daily_features(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    storage = Storage(manager.config["storage"], ROOT_DIR)
    print(
        f"Processed {len(features)} feature rows for {features['symbol'].nunique()} symbols; "
        f"saved to {storage.describe(manager.output_dir())}"
    )


if __name__ == "__main__":
    main()
