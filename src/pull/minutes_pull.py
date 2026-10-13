from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import time

import pandas as pd
import yaml

from ..data.storage import Storage
from ..data.raw.minutes_data_puller import MinutesDataPuller
from ..data.raw.ticker_renames import load_renames, ticker_for_date
from ..data.raw.util.market_calendar import is_market_session
from ..symbols.symbol_manager import SymbolManager

ROOT_DIR = Path(__file__).resolve().parents[2]
MARKET_TZ = "America/New_York"

# schedule.interval -> bar length of the matching Databento schema (ohlcv-<interval>).
BAR_LENGTHS = {
    "1s": pd.Timedelta(seconds=1),
    "1m": pd.Timedelta(minutes=1),
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
}


def load_config(path: str | Path = ROOT_DIR / "config/data/minutes_config.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    required = {"version", "provider", "schedule", "storage"}
    missing = required.difference(config or {})
    if missing:
        raise ValueError(f"minutes configuration is missing: {sorted(missing)}")
    version_config = config["version"]
    provider_config = config["provider"]
    schedule_config = config["schedule"]
    fetch_config = schedule_config.get("fetch", {})
    storage_config = config["storage"]
    if provider_config.get("provider") != "databento":
        raise ValueError("minutes provider must be 'databento'")
    api_key_env = str(provider_config.get("api_key_env", "DATABENTO_API_KEY")).strip()
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Set the {api_key_env} environment variable for the Databento API key")
    provider_config["api_key"] = api_key
    dataset = str(provider_config.get("dataset", "")).strip()
    if not dataset:
        raise ValueError("minutes databento.dataset is required")
    if not isinstance(version_config.get("data"), str) or not version_config["data"].strip():
        raise ValueError("minutes version must be a non-empty string")
    if not isinstance(version_config.get("symbols"), str) or not version_config["symbols"].strip():
        raise ValueError("minutes symbol_version must be a non-empty string")
    if not isinstance(fetch_config, dict):
        raise ValueError("minutes schedule.fetch configuration must be an object")
    has_start = bool(schedule_config.get("start_date"))
    has_end = bool(schedule_config.get("end_date"))
    if has_start != has_end:
        raise ValueError("minutes schedule requires both start_date and end_date, or neither")
    if has_start:
        start = pd.Timestamp(schedule_config["start_date"])
        end = pd.Timestamp(schedule_config["end_date"])
        if start > end:
            raise ValueError("start_date must not exceed end_date")
    interval = str(schedule_config.get("interval", "1m")).strip().lower()
    if interval not in BAR_LENGTHS:
        raise ValueError(
            f"minutes schedule.interval must be one of {sorted(BAR_LENGTHS)} "
            f"(Databento OHLCV schemas), got '{schedule_config.get('interval')}'"
        )
    schedule_config["interval"] = interval
    return config


def available_end_from_error(error: Exception) -> pd.Timestamp | None:
    message = str(error)
    match = re.search(r"available up to '([^']+)'", message) or re.search(
        r"available end of dataset [^(]+\('([^']+)'\)", message
    )
    if not match:
        return None
    available_end = pd.Timestamp(match.group(1))
    if available_end.tzinfo is None:
        return available_end.tz_localize("UTC")
    return available_end.tz_convert("UTC")


def format_day_filename(day: pd.Timestamp, interval: str, top_up: bool = False) -> str:
    """Main file, or with `top_up` a file added next to it (e.g. old tickers after a rename)."""
    return f"{day:%Y%m%d}{'.fix' if top_up else ''}.ohlcv-{interval}.dbn.zst"


def unresolved_symbols(store: object) -> list[str]:
    """Requested symbols Databento could not resolve (e.g. a ticker renamed since)."""
    metadata = getattr(store, "metadata", None)
    return sorted(getattr(metadata, "not_found", None) or [])


def freshness_summary(
    store: object,
    requested_end_utc: pd.Timestamp,
    downloaded_at_utc: pd.Timestamp,
    bar: pd.Timedelta,
    symbol_count: int,
) -> str:
    """Describe how close the downloaded bars come to the requested end time."""
    records = store.to_ndarray()
    downloaded_at = downloaded_at_utc.tz_convert(MARKET_TZ)
    if len(records) == 0:
        return f"Data check: downloaded_at={downloaded_at:%H:%M:%S} ET, rows=0"
    last_ns = records["ts_event"].max()
    last_start = pd.Timestamp(int(last_ns), tz="UTC")
    last_end = last_start + bar
    symbols_with_data = len(set(records["instrument_id"].tolist()))
    symbols_at_last = len(set(records["instrument_id"][records["ts_event"] == last_ns].tolist()))
    missing = max(requested_end_utc - last_end, pd.Timedelta(0))
    return (
        f"Data check: downloaded_at={downloaded_at:%H:%M:%S} ET, "
        f"requested_end={requested_end_utc.tz_convert(MARKET_TZ):%H:%M} ET, "
        f"last_bar={last_start.tz_convert(MARKET_TZ):%H:%M} ET ({symbols_at_last} symbols), "
        f"missing={missing.total_seconds() / 60:.0f} min, "
        f"lag={(downloaded_at_utc - last_end).total_seconds():.0f}s, "
        f"rows={len(records)}, symbols_with_data={symbols_with_data}/{symbol_count}"
    )


def session_request_window(
    day: pd.Timestamp,
    interval: str,
    session_open: str = "09:30",
    session_close: str = "16:00",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC request window for one session, widened to whole bars.

    Databento stamps each bar with its start time, so e.g. the 1h bar covering
    09:30 ET starts at 09:00 and the 1d bar starts at 00:00 UTC.
    """
    bar = BAR_LENGTHS[interval]
    session_day = pd.Timestamp(day).normalize()
    start_utc = pd.Timestamp(f"{session_day:%Y-%m-%d} {session_open}").tz_localize(
        MARKET_TZ
    ).tz_convert("UTC")
    end_utc = pd.Timestamp(f"{session_day:%Y-%m-%d} {session_close}").tz_localize(
        MARKET_TZ
    ).tz_convert("UTC")
    return start_utc.floor(bar), end_utc.ceil(bar)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull intraday stock snapshots")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "config/data/minutes_config.yaml"),
        help="Path to minute pull config YAML",
    )
    parser.add_argument(
        "--end-utc",
        help="UTC end timestamp for rolling period pulls, e.g. 2026-09-23T00:00:00Z",
    )
    parser.add_argument("--start-date", help="Start date for a range pull, e.g. 2025-01-02")
    parser.add_argument("--end-date", help="End date for a range pull, e.g. 2025-11-10")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Range pull: leave sessions whose file is already stored (resume an interrupted pull)",
    )
    parser.add_argument(
        "--renames-only",
        action="store_true",
        help="Top-up range pull: only the old tickers of renamed symbols (corporate_actions.yaml "
        "renames), for the sessions before each rename, saved as YYYYMMDD.fix.ohlcv-*.dbn.zst",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Refresh today's file with bars from the session open up to now",
    )
    return parser.parse_args()


def main(
    start_date: str | date | pd.Timestamp | None = None,
    end_date: str | date | pd.Timestamp | None = None,
    config_path: str | Path | None = None,
    end_utc: str | pd.Timestamp | None = None,
    rolling: bool = False,
    renames_only: bool = False,
    skip_existing: bool = False,
) -> None:
    config = load_config(config_path or ROOT_DIR / "config/data/minutes_config.yaml")
    version_config = config["version"]
    provider_config = config["provider"]
    schedule_config = config["schedule"]
    storage = Storage(config["storage"], ROOT_DIR)
    symbol_config = SymbolManager.load_config(str(ROOT_DIR / "config/symbols/symbols_config.yaml"))
    manager = SymbolManager(symbol_config["selection"]["random_state"])

    renames = load_renames()

    def get_puller(on_date: pd.Timestamp) -> tuple[MinutesDataPuller | None, int, pd.Timestamp]:
        """Create a puller from the latest symbol snapshot available for one session.

        Symbols are requested under the ticker valid on that session; with `renames_only`,
        only the renamed ones (no puller when the session has none).
        """
        symbol_day, symbol_data = manager.load_latest_symbol_snapshot(
            symbol_config, on_date, version=version_config["symbols"]
        )
        tickers = {
            symbol: ticker_for_date(symbol, on_date, renames) for symbol in symbol_data["symbol"]
        }
        if renames_only:
            tickers = {symbol: ticker for symbol, ticker in tickers.items() if ticker != symbol}
            if not tickers:
                return None, 0, symbol_day
        symbols = list(tickers.values())
        groups = {tickers[symbol]: group for symbol, group in zip(symbol_data["symbol"], symbol_data["index_group"]) if symbol in tickers}
        puller = MinutesDataPuller(
            symbols,
            data_path=None,
            index_groups=groups,
            provider=provider_config["provider"],
            provider_config=provider_config,
        )
        return puller, len(symbols), symbol_day

    output_dir = Path(version_config["data"])
    interval = schedule_config["interval"]
    schema = f"ohlcv-{interval}"
    bar = BAR_LENGTHS[interval]

    has_configured_range = bool(schedule_config.get("start_date") and schedule_config.get("end_date"))
    if renames_only and rolling:
        raise ValueError("--renames-only is a range pull; it cannot be combined with --rolling")
    if start_date is None and end_date is None and not renames_only and (rolling or not has_configured_range):
        now_utc = pd.Timestamp(end_utc) if end_utc is not None else pd.Timestamp.now(tz="UTC")
        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")
        else:
            now_utc = now_utc.tz_convert("UTC")
        day = now_utc.tz_convert(MARKET_TZ).tz_localize(None).normalize()
        if not is_market_session(day):
            print(f"Skip minutes pull: {day:%Y-%m-%d} is not a market session")
            return
        start_utc, close_utc = session_request_window(day, interval)
        window_end_utc = min(now_utc.floor(bar), close_utc)
        if window_end_utc <= start_utc:
            print(f"Skip minutes pull: no complete {interval} bar yet for {day:%Y-%m-%d}")
            return
        requested_end_utc = window_end_utc
        data_manager, symbol_count, symbol_day = get_puller(day)
        try:
            store = data_manager.download_range_store(start_utc, window_end_utc, schema=schema)
        except Exception as error:
            available_end = available_end_from_error(error)
            if available_end is None:
                raise
            window_end_utc = min(available_end.floor(bar), window_end_utc)
            if window_end_utc <= start_utc:
                print(f"Skip minutes pull: Databento has no {day:%Y-%m-%d} data yet")
                return
            print(
                "Retry minutes pull using Databento available window "
                f"{start_utc.isoformat()} -> {window_end_utc.isoformat()}"
            )
            store = data_manager.download_range_store(start_utc, window_end_utc, schema=schema)
        print(
            freshness_summary(
                store, requested_end_utc, pd.Timestamp.now(tz="UTC"), bar, symbol_count
            )
        )

        output_path = output_dir / format_day_filename(day, interval)
        print(f"Writing DBN+Zstd: {output_path.name}")
        write_start_clock = time.perf_counter()
        storage.write_dbn(store, output_path)
        write_elapsed = time.perf_counter() - write_start_clock
        print(
            f"Saved raw data for {symbol_count} symbols "
            f"from {symbol_day:%Y-%m-%d} to {storage.describe(output_path)} "
            f"using {provider_config['provider']}"
        )
        print(f"Write summary: {output_path.name}, write={write_elapsed:.3f}s")
        return

    if start_date is None and end_date is None:
        start_date = pd.Timestamp(schedule_config["start_date"])
        end_date = pd.Timestamp(schedule_config["end_date"])
    elif start_date is None or end_date is None:
        raise ValueError("start_date and end_date must both be provided")
    else:
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        if start_date > end_date:
            raise ValueError("start_date must not exceed end_date")

    fetch_config = schedule_config.get("fetch", {})
    max_workers = max(1, int(fetch_config.get("max_workers", 1)))

    sessions = [
        pd.Timestamp(day)
        for day in pd.date_range(start_date.normalize(), end_date.normalize(), freq="B")
        if is_market_session(day)
    ]
    if skip_existing and not renames_only:
        done = {Path(path).name for path in storage.list(output_dir, f"*.ohlcv-{interval}.dbn.zst")}
        sessions = [day for day in sessions if format_day_filename(day, interval) not in done]
    if renames_only:
        # Top-up files are only ever added, so a rerun resumes where the last one stopped.
        done = {Path(path).name for path in storage.list(output_dir, "*.fix.ohlcv-*.dbn.zst")}
        sessions = [day for day in sessions if format_day_filename(day, interval, top_up=True) not in done]
    print(
        f"Fetching {len(sessions)} session(s) with date-specific symbol files "
        f"using {max_workers} worker(s)"
    )

    def fetch_session(day: pd.Timestamp):
        puller, symbol_count, symbol_day = get_puller(day)
        if puller is None:
            return 0, symbol_day, None
        store = puller.download_range_store(*session_request_window(day, interval), schema=schema)
        return symbol_count, symbol_day, store

    saved_files = 0
    run_start_clock = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_session, day): day for day in sessions}
        for future in as_completed(futures):
            day = futures[future]
            symbol_count, symbol_day, store = future.result()
            if store is None:
                continue
            write_start_clock = time.perf_counter()
            output_path = output_dir / format_day_filename(day, interval, top_up=renames_only)
            storage.write_dbn(store, output_path)
            write_elapsed = time.perf_counter() - write_start_clock
            saved_files += 1
            unresolved = unresolved_symbols(store)
            print(
                f"Session summary: {day:%Y-%m-%d}, file={output_path.name}, "
                f"symbols={symbol_count}, symbol_date={symbol_day:%Y-%m-%d}, "
                f"write={write_elapsed:.3f}s"
                + (f", unresolved={','.join(unresolved)} (renamed? add to corporate_actions.yaml renames)" if unresolved else "")
            )

    print(
        f"Saved {saved_files} file(s) "
        f"with date-specific symbols to {storage.describe(output_dir)} "
        f"in {time.perf_counter() - run_start_clock:.1f}s"
    )

if __name__ == "__main__":
    args = parse_args()
    main(
        args.start_date,
        args.end_date,
        config_path=args.config,
        end_utc=args.end_utc,
        rolling=args.rolling,
        renames_only=args.renames_only,
        skip_existing=args.skip_existing,
    )
