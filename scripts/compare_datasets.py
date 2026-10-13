#!/usr/bin/env python3
"""Compare two Databento datasets' 1m bars (default XNAS.BASIC vs EQUS.MINI) on sample days.

For each pair of consecutive sessions (D, D+1), over the regular session and the day's
symbol list:

* coverage: share of (symbol, minute) bars present, share of symbols with the bar before
  the cutoff (15:57 for a 15:58 cutoff);
* volume: the second dataset's session volume as a share of the first's (per symbol,
  median), to see whether the level is stable over time;
* prices: 15:57 close and the 09:30 open, differences in basis points;
* overnight return (next open / 15:57 close - 1, the label's return): correlation and
  typical difference between the two datasets.

Bars are downloaded to a local cache (the scratch directory given by --cache); nothing is
written to the pipeline's storage.

    python3 scripts/compare_datasets.py --cache /tmp/compare
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.raw.dbn_store import load_dbn_bars  # noqa: E402
from src.data.raw.ticker_renames import load_renames, ticker_for_date  # noqa: E402
from src.data.raw.util.market_calendar import market_sessions  # noqa: E402
from src.data.raw.util.normalize import preprocess_symbol  # noqa: E402
from src.symbols.symbol_manager import SymbolManager  # noqa: E402

MARKET_TZ = "America/New_York"
PAIRS = ["2024-08-13", "2024-12-10", "2025-03-12", "2025-06-10", "2025-09-16", "2026-01-13", "2026-04-14", "2026-09-28"]


def symbols_for(day: pd.Timestamp) -> list[str]:
    """Pipeline symbols of the day's snapshot (the earliest one for days before the snapshots start)."""
    config = SymbolManager.load_config(str(ROOT_DIR / "config/symbols/symbols_config.yaml"))
    manager = SymbolManager(config["selection"]["random_state"])
    _, snapshot = manager.load_latest_symbol_snapshot(config, max(day, pd.Timestamp("2025-01-01")), version="v1")
    return sorted(snapshot["symbol"])


def session_bars(client, dataset: str, day: pd.Timestamp, symbols: list[str], cache: Path) -> pd.DataFrame:
    """Regular-session 1m bars under pipeline symbols (renamed tickers mapped back)."""
    path = cache / f"{dataset}.{day:%Y%m%d}.dbn.zst"
    renames = load_renames()
    tickers = {preprocess_symbol(ticker_for_date(symbol, day, renames)): symbol for symbol in symbols}
    if not path.exists():
        start = pd.Timestamp(f"{day:%Y-%m-%d} 09:30", tz=MARKET_TZ).tz_convert("UTC")
        end = pd.Timestamp(f"{day:%Y-%m-%d} 16:00", tz=MARKET_TZ).tz_convert("UTC")
        store = client.timeseries.get_range(dataset=dataset, schema="ohlcv-1m", symbols=sorted(tickers), start=start, end=end)
        store.to_file(path, compression="zstd")
    bars = load_dbn_bars([path], adjust=False)
    provider = bars["symbol"].str.replace(r"-([A-Z])$", r".\1", regex=True)
    bars["symbol"] = provider.map(tickers).fillna(bars["symbol"])
    return bars[bars["symbol"].isin(symbols)]


def compare_pair(client, first_ds: str, second_ds: str, day: pd.Timestamp, cutoff: str, cache: Path) -> dict:
    next_day = [s for s in market_sessions(day, day + pd.Timedelta(days=10)) if s > day][0]
    symbols = symbols_for(day)
    a, b = (session_bars(client, ds, day, symbols, cache) for ds in (first_ds, second_ds))
    a_next, b_next = (session_bars(client, ds, next_day, symbols, cache) for ds in (first_ds, second_ds))
    minutes = 390
    last_bar = pd.Timestamp(f"{day:%Y-%m-%d} {cutoff}", tz=MARKET_TZ).tz_convert("UTC") - pd.Timedelta(minutes=1)

    def at(bars, time):
        return bars[bars["datetime_utc"] == time].set_index("symbol")

    def opens(bars):
        return bars.sort_values("datetime_utc").groupby("symbol")["open"].first()

    volume = (b.groupby("symbol")["volume"].sum() / a.groupby("symbol")["volume"].sum()).replace([np.inf, -np.inf], np.nan)
    close_a, close_b = at(a, last_bar)["close"], at(b, last_bar)["close"]
    both = close_a.index.intersection(close_b.index)
    close_bp = ((close_b[both] / close_a[both] - 1) * 1e4).abs()
    open_a, open_b = opens(a_next), opens(b_next)
    both_open = open_a.index.intersection(open_b.index)
    open_bp = ((open_b[both_open] / open_a[both_open] - 1) * 1e4).abs()
    ret_a = (open_a / close_a - 1).dropna()
    ret_b = (open_b / close_b - 1).dropna()
    common = ret_a.index.intersection(ret_b.index)
    return {
        "day": f"{day:%Y-%m-%d}",
        f"{first_ds} bars": a.groupby("symbol").size().sum() / (len(symbols) * minutes),
        f"{second_ds} bars": b.groupby("symbol").size().sum() / (len(symbols) * minutes),
        f"{first_ds} last bar": close_a.size / len(symbols),
        f"{second_ds} last bar": close_b.size / len(symbols),
        "volume share (median)": volume.median(),
        "close diff bp (median)": close_bp.median(),
        "close diff >5bp": (close_bp > 5).mean(),
        "open diff bp (median)": open_bp.median(),
        "open diff >5bp": (open_bp > 5).mean(),
        "overnight corr": ret_a[common].corr(ret_b[common]),
        "overnight rank corr": ret_a[common].corr(ret_b[common], method="spearman"),
        "overnight diff bp (median)": ((ret_b[common] - ret_a[common]).abs() * 1e4).median(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="Local directory for the downloaded bars")
    parser.add_argument("--first", default="XNAS.BASIC")
    parser.add_argument("--second", default="EQUS.MINI")
    parser.add_argument("--cutoff", default="15:58", help="Decision time; the bar before it is compared")
    parser.add_argument("--days", nargs="*", default=PAIRS, help="First day of each (D, D+1) pair")
    args = parser.parse_args()
    import databento

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    client = databento.Historical(os.environ["DATABENTO_API_KEY"])
    rows = []
    for day in args.days:
        rows.append(compare_pair(client, args.first, args.second, pd.Timestamp(day), args.cutoff, cache))
        print(f"  {day} done", flush=True)
    table = pd.DataFrame(rows).set_index("day")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(table.T.to_string())


if __name__ == "__main__":
    main()
