"""Build regular-session daily bars from the raw 1m files (see src/data/raw/session_bars.py).

    python -m src.pull.session_bars                     # schedule.start_date..end_date
    python -m src.pull.session_bars --date 2026-09-25   # one day, e.g. after the 16:15 pull
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from ..data.raw.dbn_store import files_by_day, load_stored_dbn_bars
from ..data.raw.session_bars import session_daily_bars, session_filename
from ..data.storage import Storage

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT_DIR / "config/data/session_bars_config.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    missing = {"version", "schedule", "storage"}.difference(config or {})
    if missing:
        raise ValueError(f"session bars configuration is missing: {sorted(missing)}")
    for section, name in (("input", "raw_minutes"), ("output", "session_bars")):
        if not config["storage"].get(section, {}).get(name):
            raise ValueError(f"session bars storage.{section}.{name} is required")
    return config


def main(start_date: str | None = None, end_date: str | None = None, config_path: str | Path | None = None) -> None:
    config = load_config(config_path) if config_path else load_config()
    schedule = config["schedule"]
    start = pd.Timestamp(start_date or schedule["start_date"]).normalize()
    end = pd.Timestamp(end_date or schedule["end_date"]).normalize()
    storage = Storage(config["storage"], ROOT_DIR)
    if storage.backend == "s3":
        _ = storage.client  # create the shared S3 client before worker threads use it
    input_dir = Path(config["storage"]["input"]["raw_minutes"]) / config["version"]["raw_minutes"]
    output_dir = Path(config["storage"]["output"]["session_bars"]) / config["version"]["data"]
    files = files_by_day(storage.list(input_dir, "*.ohlcv-1m.dbn.zst"))
    days = sorted(day for day in files if start <= day <= end)
    if not days:
        raise FileNotFoundError(f"No raw minute files between {start.date()} and {end.date()} in {storage.describe(input_dir)}")
    print(f"Building session bars for {len(days)} day(s) from {storage.describe(input_dir)}")

    def build(day: pd.Timestamp) -> int:
        minute_bars = load_stored_dbn_bars(storage, files[day], adjust=False)
        bars = session_daily_bars(minute_bars)
        storage.write_parquet(bars, output_dir / session_filename(day))
        return len(bars)

    started = time.perf_counter()
    workers = max(1, int(schedule.get("max_workers", 1)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build, day): day for day in days}
        for done, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            if done % 50 == 0 or done == len(days):
                print(f"  {done}/{len(days)} days, last {futures[future]:%Y-%m-%d}: {rows} symbols")
    print(f"Saved {len(days)} file(s) to {storage.describe(output_dir)} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build regular-session daily bars from 1m bars")
    parser.add_argument("--date", help="Build a single day, e.g. 2026-09-25 (default: schedule range)")
    args = parser.parse_args()
    main(args.date, args.date)
