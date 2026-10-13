"""Today's 1m bars up to the cutoff from Databento's Live API.

The historical API publishes same-day bars about 10 minutes late, so at the cutoff it
cannot supply the last bars. The Live API sends each 1m bar when its minute ends, and
`start` (intraday replay) first replays the session from the open, so one connection
opened shortly before the cutoff delivers every bar from 09:30 up to the cutoff.

The stream is saved as the day's raw minute file (same dataset, schema and location as
the historical pull), so features are computed by the unchanged batch code. The nightly
job later replaces the file with the full session from the historical API.

    python -m src.pull.live_bars                     # today, up to the configured cutoff
    python -m src.pull.live_bars --date 2026-09-28   # replay a session from the last 24h (test)
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd

from ..data.features.minutes_feature_manager import MinutesFeatureManager, session_bounds
from ..data.raw.ticker_renames import load_renames, ticker_for_date
from ..data.raw.util.normalize import preprocess_symbol
from ..data.storage import Storage
from ..symbols.symbol_manager import SymbolManager
from . import minutes_pull

ROOT_DIR = Path(__file__).resolve().parents[2]
MARKET_TZ = "America/New_York"


def session_tickers(day: pd.Timestamp, config: dict) -> list[str]:
    """Provider tickers for the day's symbol snapshot (renames applied, BRK-B -> BRK.B)."""
    symbol_config = SymbolManager.load_config(str(ROOT_DIR / "config/symbols/symbols_config.yaml"))
    manager = SymbolManager(symbol_config["selection"]["random_state"])
    _, snapshot = manager.load_latest_symbol_snapshot(symbol_config, day, version=config["version"]["symbols"])
    renames = load_renames()
    return sorted({preprocess_symbol(ticker_for_date(symbol, day, renames)) for symbol in snapshot["symbol"]})


def stream_until_cutoff(
    day: pd.Timestamp,
    cutoff_utc: pd.Timestamp,
    grace_seconds: float = 3.0,
    max_wait_seconds: float = 45.0,
    config_path: str | Path | None = None,
) -> dict:
    """Stream the day's 1m bars from the open until the bar ending at `cutoff_utc`; save the raw file.

    Stops when a bar at or after the cutoff arrives (replaying a past session), or once the
    wall clock is `grace_seconds` past the cutoff and the last bar has arrived, or at the
    latest `max_wait_seconds` after the cutoff. Returns counts for the freshness check.
    """
    databento = importlib.import_module("databento")
    config = minutes_pull.load_config(config_path or ROOT_DIR / "config/data/minutes_config.yaml")
    provider = config["provider"]
    interval = config["schedule"]["interval"]
    if interval != "1m":
        raise ValueError(f"live bars need schedule.interval 1m, got {interval}")
    tickers = session_tickers(day, config)
    open_utc = session_bounds(day, 0)[0]
    last_bar_ns = (cutoff_utc - pd.Timedelta(minutes=1)).value
    cutoff_ns = cutoff_utc.value

    state = {"bars": 0, "last_bar": set(), "instruments": set(), "latest_ns": 0, "errors": []}
    reached = threading.Event()

    def on_record(record) -> None:
        if isinstance(record, databento.SymbolMappingMsg):
            state["instruments"].add(record.instrument_id)
        elif isinstance(record, databento.OHLCVMsg):
            state["bars"] += 1
            state["latest_ns"] = max(state["latest_ns"], record.ts_event)
            if record.ts_event == last_bar_ns:
                state["last_bar"].add(record.instrument_id)
            if record.ts_event >= cutoff_ns:
                reached.set()
        elif isinstance(record, databento.ErrorMsg):
            state["errors"].append(record.err)

    with tempfile.TemporaryDirectory() as directory:
        stream_path = Path(directory) / f"{day:%Y%m%d}.live.dbn"
        client = databento.Live(key=provider["api_key"])
        client.subscribe(
            dataset=provider["dataset"],
            schema=f"ohlcv-{interval}",
            symbols=tickers,
            stype_in=provider.get("stype_in", "raw_symbol"),
            start=open_utc,
        )
        client.add_stream(stream_path)
        client.add_callback(on_record)
        started = time.perf_counter()
        client.start()
        # A replayed past session is past its cutoff already: give the replay time to arrive.
        deadline = max(cutoff_utc + pd.Timedelta(seconds=max_wait_seconds), pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=2))
        while not reached.is_set():
            now = pd.Timestamp.now(tz="UTC")
            if now >= deadline:
                break
            if now >= cutoff_utc + pd.Timedelta(seconds=grace_seconds) and state["latest_ns"] >= last_bar_ns:
                break
            reached.wait(0.2)
        client.stop()
        client.block_for_close(timeout=10)
        store = databento.DBNStore.from_file(stream_path)
        storage = Storage(config["storage"], ROOT_DIR)
        output = Path(config["version"]["data"]) / minutes_pull.format_day_filename(day, interval)
        storage.write_dbn(store, output)

    summary = {
        "tickers": len(tickers),
        "instruments": len(state["instruments"]),
        "bars": state["bars"],
        "with_last_bar": len(state["last_bar"]),
        "latest_bar": pd.Timestamp(state["latest_ns"], tz="UTC").tz_convert(MARKET_TZ) if state["latest_ns"] else None,
        "seconds": round(time.perf_counter() - started, 1),
        "errors": state["errors"],
        "path": storage.describe(output),
    }
    print(
        f"Live bars {day:%Y-%m-%d}: {summary['bars']} bars for {summary['instruments']}/{summary['tickers']} tickers, "
        f"{summary['with_last_bar']} with the {pd.Timestamp(last_bar_ns, tz='UTC').tz_convert(MARKET_TZ):%H:%M} bar, "
        f"latest {summary['latest_bar']:%H:%M} in {summary['seconds']}s; saved {summary['path']}"
        + (f"; gateway errors: {summary['errors']}" if summary["errors"] else "")
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream today's 1m bars up to the cutoff (Databento Live)")
    parser.add_argument("--date", help="Session to stream (default: today); past sessions replay if within ~24h")
    parser.add_argument("--config", default=str(ROOT_DIR / "config/data/minutes_config.yaml"),
                        help="Minute pull config: dataset and where the file is saved (minutes_equs_mini_config.yaml for EQUS.MINI)")
    args = parser.parse_args()
    day = pd.Timestamp(args.date) if args.date else pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None).normalize()
    minutes = MinutesFeatureManager(ROOT_DIR / "config/feature/minutes_feature_config.yaml")
    stream_until_cutoff(day, session_bounds(day, minutes.cutoff_before_close)[2], config_path=args.config)


if __name__ == "__main__":
    main()
