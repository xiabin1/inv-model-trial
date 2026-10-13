from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from ..data.storage import Storage
from ..data.raw.daily_data_puller import DailyDataPuller
from ..data.raw.ticker_renames import load_renames, ticker_for_date
from ..data.raw.util.market_calendar import is_market_session
from ..symbols.symbol_manager import SymbolManager

ROOT_DIR = Path(__file__).resolve().parents[2]


def previous_weekday() -> date:
    day = date.today() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def load_config(path: str | Path = ROOT_DIR / "config/data/daily_config.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    required = {"version", "provider", "schedule", "storage"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"daily configuration is missing: {sorted(missing)}")

    version_config = config["version"]
    provider_config = config["provider"]
    schedule_config = config["schedule"]
    storage_config = config["storage"]
    if not all(isinstance(value, dict) for value in (version_config, provider_config, schedule_config)):
        raise ValueError("daily version, provider, and schedule configuration must be objects")
    if not isinstance(storage_config, dict):
        raise ValueError("daily storage configuration must be an object")

    storage_backend = storage_config.get("backend", "local")
    if storage_backend not in {"local", "s3"}:
        raise ValueError("daily storage.backend must be 'local' or 's3'")
    if provider_config.get("provider") != "databento":
        raise ValueError("daily provider must be 'databento'")
    api_key_env = str(provider_config.get("api_key_env", "DATABENTO_API_KEY")).strip()
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"Set the {api_key_env} environment variable for the Databento API key"
        )
    provider_config["api_key"] = api_key
    if not str(provider_config.get("dataset", "")).strip():
        raise ValueError("daily databento.dataset is required")
    if not isinstance(version_config.get("data"), str) or not version_config["data"].strip():
        raise ValueError("daily version must be a non-empty string")
    if not isinstance(version_config.get("symbols"), str) or not version_config["symbols"].strip():
        raise ValueError("daily symbol_version must be a non-empty string")
    has_start = bool(schedule_config.get("start_date"))
    has_end = bool(schedule_config.get("end_date"))
    if has_start != has_end:
        raise ValueError("daily schedule requires both start_date and end_date, or neither")
    if has_start and schedule_config["start_date"] > schedule_config["end_date"]:
        raise ValueError("start_date must not exceed end_date")
    if not isinstance(schedule_config.get("fetch", {}), dict):
        raise ValueError("daily schedule.fetch configuration must be an object")
    return config


def daily_filename(day: pd.Timestamp) -> str:
    return f"{day:%Y%m%d}.ohlcv-1d.dbn.zst"


def save_daily_store(
    store: object,
    storage: Storage,
    output_dir: str,
    day: pd.Timestamp,
) -> str:
    """Save one session's daily bars as a Zstd-compressed DBN file."""
    path = Path(output_dir) / daily_filename(day)
    storage.write_dbn(store, path)
    return storage.describe(path)


def main(
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
) -> None:
    config = load_config()
    storage = Storage(config["storage"], ROOT_DIR)
    version_config = config["version"]
    provider_config = config["provider"]
    schedule_config = config["schedule"]
    symbol_config = SymbolManager.load_config(str(ROOT_DIR / "config/symbols/symbols_config.yaml"))

    if start_date is None and end_date is None:
        if schedule_config.get("start_date") and schedule_config.get("end_date"):
            start_date = pd.Timestamp(schedule_config["start_date"])
            end_date = pd.Timestamp(schedule_config["end_date"])
        else:
            start_date = end_date = pd.Timestamp(previous_weekday())
    else:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date must both be provided")
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        if start_date > end_date:
            raise ValueError("start_date must not exceed end_date")

    output_dir = str(Path(version_config["data"]))
    fetch_config = schedule_config.get("fetch", {})
    max_workers = max(1, int(fetch_config.get("max_workers", 1)))
    sessions = [
        pd.Timestamp(day)
        for day in pd.date_range(start_date, end_date, freq="B")
        if is_market_session(day)
    ]
    manager = SymbolManager(symbol_config["selection"]["random_state"])
    renames = load_renames()

    def fetch_session(day: pd.Timestamp):
        symbol_data = manager.load_symbol_snapshot(
            symbol_config, day, version=version_config["symbols"]
        )
        # Request each symbol under the ticker it traded as that day.
        symbols = [ticker_for_date(symbol, day, renames) for symbol in symbol_data["symbol"]]
        groups = dict(zip(symbols, symbol_data["index_group"]))
        puller = DailyDataPuller(
            symbols,
            data_path=None,
            index_groups=groups,
            provider=provider_config["provider"],
            provider_config=provider_config,
        )
        store = puller.download_range_store(
            day.strftime("%Y-%m-%d"),
            (day + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
        )
        return day, len(symbols), store

    print(f"Fetching {len(sessions)} session(s) using {max_workers} worker(s)")
    run_start_clock = time.perf_counter()
    saved_files = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch_session, day) for day in sessions]
        for future in as_completed(futures):
            day, symbol_count, store = future.result()
            output_path = save_daily_store(store, storage, output_dir, day)
            saved_files += 1
            print(
                f"Saved daily DBN data for {symbol_count} symbols to {output_path} "
                "using Zstd compression"
            )
    print(
        f"Completed {saved_files} session(s) in {time.perf_counter() - run_start_clock:.1f}s"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pull daily stock data")
    parser.add_argument("--start-date", help="Start date, e.g. 2026-09-20 or 20260920")
    parser.add_argument("--end-date", help="End date, e.g. 2026-09-20 or 20260920")
    args = parser.parse_args()
    main(args.start_date, args.end_date)
