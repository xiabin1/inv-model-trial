"""Daily stock lists of the b1542_l250 profile: S&P 500 / Nasdaq-100 members plus liquid US stocks.

Rules for day D, using only sessions before D (EQUS.SUMMARY daily bars, all exchanges):
  * common stock or ADR (Databento security_type C / A), last close >= $5;
  * median daily dollar volume (close x volume) over the last 20 sessions:
      enter at >= $250M, leave below $200M, and only after 15 of the last 20 sessions at >= $125M
      (so the minute history the features need already exists);
  * plus every S&P 500 / Nasdaq-100 member (symbols/v1).
Lists (same CSV format as symbols/v1, index_group "liquid" for the added stocks):
  symbols/l250_v1/D.csv       training universe (features, labels)
  symbols/l250_v1_pull/D.csv  minute download: the list plus stocks at >= $125M (on their way in)
Raw data: raw/daily_all/v1/YYYYMMDD.parquet (one row per symbol) and raw/daily_all/v1/types.parquet.
Rules and thresholds come from scripts/experiments/liquid250.py (tested 2025-06..2026-10).

    python -m src.symbols.liquidity_universe update --date 2026-10-08   # pull the session, build the next lists
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import yaml

from ..data.raw.util.market_calendar import market_sessions
from ..data.storage import Storage

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_CONFIG = ROOT_DIR / "config/data/session_bars_config.yaml"  # storage of financial/tp1/data
SYMBOLS_CONFIG = ROOT_DIR / "config/symbols/symbols_config.yaml"
RAW_DIR = Path("raw/daily_all/v1")
LIST, PULL_LIST, INDEX_LIST = "l250_v1", "l250_v1_pull", "v1"
ENTER, LEAVE, ON_WAY, MIN_PRICE, WINDOW, MIN_DAYS = 250e6, 200e6, 125e6, 5.0, 20, 15
STOCK_TYPES = {"C", "A"}


def storage() -> Storage:
    return Storage(yaml.safe_load(DATA_CONFIG.read_text())["storage"], ROOT_DIR)


def symbols_storage() -> tuple[Storage, Path]:
    config = yaml.safe_load(SYMBOLS_CONFIG.read_text())
    return Storage(config["storage"], ROOT_DIR), Path(".")


def ticker(symbol: str) -> str:
    return str(symbol).replace(".", "-")  # Databento BRK.B -> BRK-B as in symbols/v1


# ---------------------------------------------------------------- raw data

def retried(call, attempts: int = 4, wait: float = 30.0):
    """Databento's gateway sometimes answers 504; try again a few times."""
    import time

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:  # BentoServerError and network errors
            if attempt == attempts:
                raise
            print(f"Databento: {type(error).__name__} ({str(error)[:80]}); retry {attempt}/{attempts - 1} in {wait:.0f}s")
            time.sleep(wait)


def pull_daily_all(day: pd.Timestamp) -> int:
    """EQUS.SUMMARY daily bars of every symbol for one session, with tickers resolved."""
    import databento as db

    day = pd.Timestamp(day).normalize()
    client = db.Historical()
    end = day + pd.Timedelta(days=1)
    frame = retried(lambda: client.timeseries.get_range(dataset="EQUS.SUMMARY", schema="ohlcv-1d", symbols="ALL_SYMBOLS",
                                                        start=day, end=end).to_df().reset_index())
    if frame.empty:
        raise RuntimeError(f"No EQUS.SUMMARY daily bars for {day:%Y-%m-%d} yet")
    ids = sorted(frame["instrument_id"].unique())
    names = {}
    for i in range(0, len(ids), 2000):
        chunk = [str(x) for x in ids[i:i + 2000]]
        result = retried(lambda: client.symbology.resolve(dataset="EQUS.SUMMARY", symbols=chunk, stype_in="instrument_id",
                                                          stype_out="raw_symbol", start_date=f"{day:%Y-%m-%d}",
                                                          end_date=f"{end:%Y-%m-%d}"))["result"]
        names.update({int(k): v[0]["s"] for k, v in result.items() if v})
    frame["ticker"] = frame["instrument_id"].map(names)
    frame["day"] = day
    frame = frame.dropna(subset=["ticker"])[["day", "instrument_id", "ticker", "open", "high", "low", "close", "volume"]]
    storage().write_parquet(frame, RAW_DIR / f"{day:%Y%m%d}.parquet")
    update_types(day, client)
    update_first_seen(frame)
    return len(frame)


def update_first_seen(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """First session each ticker traded (raw/daily_all/v1/first_seen.parquet), for the listing-age rule.

    Without `frame`, rebuilt from every stored daily file; with `frame` (one new session), only new
    tickers are added. Tickers already trading on the first stored session (2024-09-03) are old."""
    store = storage()
    path = RAW_DIR / "first_seen.parquet"
    if frame is None or not store.exists(path):
        days = sorted(p for p in store.list(RAW_DIR, "*.parquet") if Path(p).stem.isdigit())
        seen: dict[str, pd.Timestamp] = {}
        for name in days:
            for symbol in store.read_parquet(RAW_DIR / Path(name).name)["ticker"].unique():
                seen.setdefault(ticker(symbol), pd.Timestamp(Path(name).stem))
        table = pd.DataFrame({"symbol": list(seen), "first_day": list(seen.values())})
    else:
        table = store.read_parquet(path)
        new = sorted({ticker(s) for s in frame["ticker"]} - set(table["symbol"]))
        table = pd.concat([table, pd.DataFrame({"symbol": new, "first_day": pd.Timestamp(frame["day"].iloc[0])})])
    store.write_parquet(table, path)
    return table


def update_types(day: pd.Timestamp, client=None) -> None:
    """Security type per ticker (common, ADR, ETF, ...), latest definition wins."""
    import databento as db

    client = client or db.Historical()
    store = storage()
    found = retried(lambda: client.timeseries.get_range(dataset="EQUS.SUMMARY", schema="definition", symbols="ALL_SYMBOLS",
                                                        start=day, end=day + pd.Timedelta(days=1)).to_df())
    if found.empty:
        return
    new = found[["raw_symbol", "security_type"]].assign(asof=pd.Timestamp(day))
    path = RAW_DIR / "types.parquet"
    old = store.read_parquet(path) if store.exists(path) else new.iloc[0:0]
    types = pd.concat([old, new]).sort_values("asof").drop_duplicates("raw_symbol", keep="last")
    store.write_parquet(types, path)


# ---------------------------------------------------------------- lists

def select(bars: pd.DataFrame, types: pd.Series, previous_liquid: set[str]) -> tuple[set[str], set[str]]:
    """(liquid stocks, stocks on their way in) for the session after the last one in `bars`.

    `bars`: day, ticker, close, volume for at least the last 40 sessions before the list's day."""
    bars = bars[bars["ticker"].map(types).isin(STOCK_TYPES)].copy()
    bars["dv"] = bars["close"] * bars["volume"]
    dv = bars.pivot_table(index="day", columns="ticker", values="dv", aggfunc="sum").sort_index()
    close = bars.pivot_table(index="day", columns="ticker", values="close", aggfunc="last").reindex_like(dv)
    median = dv.rolling(WINDOW, min_periods=1).median()  # row i: sessions up to i, i.e. known for the next day
    last_close = close.ffill()
    on_way = (median >= ON_WAY) & (last_close >= MIN_PRICE)
    days_on_way = on_way.astype(int).rolling(WINDOW, min_periods=1).sum()
    m, p, n = median.iloc[-1], last_close.iloc[-1], days_on_way.iloc[-2] if len(days_on_way) > 1 else days_on_way.iloc[-1] * 0
    enter = (m >= ENTER) & (p >= MIN_PRICE) & (n >= MIN_DAYS)
    stay = (m >= LEAVE) & (p >= MIN_PRICE)
    previous = pd.Series(dv.columns.isin([s.replace("-", ".") for s in previous_liquid]), index=dv.columns)
    liquid = (previous & stay) | enter
    return ({ticker(s) for s in liquid.index[liquid.to_numpy()]},
            {ticker(s) for s in on_way.columns[on_way.iloc[-1].to_numpy()]})


def latest_list(name: str, day: pd.Timestamp, before: bool = False) -> pd.DataFrame | None:
    store, _ = symbols_storage()
    files = sorted(store.list(Path(name), "*.csv"))
    dated = [(pd.Timestamp(Path(f).stem), f) for f in files]
    dated = [(d, f) for d, f in dated if (d < day if before else d <= day)]
    return store.read_csv(Path(name) / Path(dated[-1][1]).name) if dated else None


def build_lists(day: pd.Timestamp) -> dict:
    """Write symbols/l250_v1/D.csv and symbols/l250_v1_pull/D.csv for session `day`."""
    day = pd.Timestamp(day).normalize()
    store = storage()
    sessions = [s for s in market_sessions(day - pd.Timedelta(days=90), day) if s < day][-(2 * WINDOW + 5):]
    paths = [RAW_DIR / f"{s:%Y%m%d}.parquet" for s in sessions]
    missing = [p.name for p in paths if not store.exists(p)]
    if missing:
        raise FileNotFoundError(f"daily_all files missing for the {day:%Y-%m-%d} lists: {missing[-5:]}")
    bars = pd.concat(store.read_parquet(p)[["day", "ticker", "close", "volume"]] for p in paths)
    types = store.read_parquet(RAW_DIR / "types.parquet").set_index("raw_symbol")["security_type"]
    previous = latest_list(LIST, day, before=True)
    previous_liquid = set(previous.loc[previous["index_group"] == "liquid", "symbol"]) if previous is not None else set()
    liquid, on_way = select(bars, types, previous_liquid)
    members = latest_list(INDEX_LIST, day)
    if members is None:
        raise FileNotFoundError(f"No symbols/{INDEX_LIST} snapshot on or before {day:%Y-%m-%d}")
    members = members.drop_duplicates("symbol")[["symbol", "index_group", "company_name"]]
    added = sorted(liquid - set(members["symbol"]))
    frame = pd.concat([members, pd.DataFrame({"symbol": added, "index_group": "liquid", "company_name": "Unknown"})])
    frame.insert(0, "date", f"{day:%Y-%m-%d}")
    pull = pd.concat([frame, pd.DataFrame({"date": f"{day:%Y-%m-%d}", "symbol": sorted(on_way - set(frame["symbol"])),
                                           "index_group": "liquid", "company_name": "Unknown"})])
    symbol_store, _ = symbols_storage()
    for name, data in ((LIST, frame), (PULL_LIST, pull)):
        symbol_store.write_bytes(Path(name) / f"{day:%Y-%m-%d}.csv", data.to_csv(index=False).encode(), "text/csv")
    joined = sorted(set(added) - previous_liquid)
    left = sorted(previous_liquid - set(added))
    print(f"l250 lists {day:%Y-%m-%d}: {len(frame)} stocks ({len(members)} index + {len(added)} liquid), "
          f"download {len(pull)}; joined {joined or '-'}, left {left or '-'}")
    return {"stocks": len(frame), "liquid": len(added), "download": len(pull), "joined": joined, "left": left}


def update(day: pd.Timestamp) -> None:
    """Nightly: store session `day`'s daily bars, then build the lists of the next session."""
    rows = pull_daily_all(day)
    print(f"daily_all {pd.Timestamp(day):%Y-%m-%d}: {rows} symbols")
    nxt = next(s for s in market_sessions(pd.Timestamp(day) + pd.Timedelta(days=1), pd.Timestamp(day) + pd.Timedelta(days=10)))
    build_lists(nxt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily lists of the b1542_l250 universe")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("update", "build"):
        command = commands.add_parser(name)
        command.add_argument("--date", required=True)
    args = parser.parse_args()
    if args.command == "update":
        update(pd.Timestamp(args.date))
    else:
        build_lists(pd.Timestamp(args.date))


if __name__ == "__main__":
    main()
