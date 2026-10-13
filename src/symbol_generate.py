from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from .symbols.symbol_manager import SymbolManager


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config/symbols/symbols_config.yaml"


def main(
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
) -> None:
    config = SymbolManager.load_config(str(CONFIG_PATH))
    manager = SymbolManager(config["selection"]["random_state"])

    if start_date is None and end_date is None:
        schedule = config.get("schedule", {})
        if not schedule.get("start_date") or not schedule.get("end_date"):
            raise ValueError(
                "Provide start_date and end_date, or configure schedule.start_date and schedule.end_date"
            )
        start_date = schedule["start_date"]
        end_date = schedule["end_date"]
    elif start_date is None or end_date is None:
        raise ValueError("start_date and end_date must both be provided")

    if str(start_date) == str(end_date):
        result = manager.generate_symbols(config, start_date)
        output_path = manager.symbol_snapshot_path(
            manager.storage_base_dir(config), config["version"]["data"], start_date
        )
        print(f"Generated {len(result)} symbols in {output_path}")
        return

    snapshots = manager.generate_symbol_snapshots(config, start_date, end_date)
    output_dir = Path(manager.storage_base_dir(config)) / config["version"]["data"]
    print(f"Generated {len(snapshots)} daily symbol files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate point-in-time symbol files")
    parser.add_argument("--start-date", help="Start date, e.g. 2026-09-20 or 20260920")
    parser.add_argument("--end-date", help="End date, e.g. 2026-09-20 or 20260920")
    args = parser.parse_args()
    main(args.start_date, args.end_date)
