"""Paper trading: mock-buy the day's top picks at the cutoff, mock-sell at the next open.

Ledgers are CSV files under data/paper/ (`live_trades.csv` for the scheduled run,
`sim_trades.csv` for `simulate`), one row per trade:

* recorded at the prediction: trade date, rank, symbol, score, model, cutoff price
  (the last 1m close the model saw) and the decision time;
* `buy_price`: the VWAP of the 1m bar starting at the cutoff (15:58-15:59), i.e. the
  minute in which the order would be filled. Filled by `settle` once the full-session
  file exists (nightly run), since that bar is not complete at prediction time;
* `sell_price`: the next session's 09:30 opening trade, as in the model's label;
* returns before and after the round-trip cost, shares and P&L for a fixed amount per stock.

    python -m src.paper settle                               # fill prices of open live trades
    python -m src.paper simulate --start 2026-09-14 --end 2026-09-25
    python -m src.paper report                               # summary of the live ledger
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data.features.minutes_feature_manager import MinutesFeatureManager, session_bounds
from .data.raw.dbn_store import files_by_day, load_stored_dbn_bars
from .data.raw.session_bars import load_session_bars, session_filename
from .data.raw.ticker_renames import load_renames, ticker_for_date
from .data.raw.util.normalize import preprocess_symbol
from .data.raw.util.market_calendar import is_early_close, market_sessions
from .data.storage import Storage
from .pull import minutes_pull, session_bars
from .train import ROOT_DIR, load_runtime_config

MARKET_TZ = "America/New_York"
LEDGER_DIR = ROOT_DIR / "data/paper"
SAME_DAY_DELAY_MINUTES = 15  # same-day historical XNAS.BASIC lag on the plan (measured 2026-10-02)
COLUMNS = [
    "trade_date", "rank", "symbol", "score", "model_trained_to", "decided_at", "cutoff_price",
    "buy_price", "sell_date", "sell_price", "shares", "gross_return", "net_return", "pnl", "status",
]


def amount_per_stock(paper: dict) -> float:
    """Dollars per pick: the day's capital split equally across the `top_n` picks."""
    return float(paper["capital_per_day"]) / int(paper["top_n"])


def ledger_path(ledger: str) -> Path:
    return LEDGER_DIR / f"{ledger}_trades.csv"


def load_ledger(ledger: str) -> pd.DataFrame:
    path = ledger_path(ledger)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    data = pd.read_csv(path, dtype={"symbol": str})
    for column in ("trade_date", "sell_date"):
        data[column] = pd.to_datetime(data[column])
    return data


def save_ledger(data: pd.DataFrame, ledger: str) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    data = data[COLUMNS].sort_values(["trade_date", "rank"])
    data.to_csv(ledger_path(ledger), index=False, date_format="%Y-%m-%d")


def record_buys(top: pd.DataFrame, day: pd.Timestamp, model_trained_to: str, ledger: str) -> pd.DataFrame:
    """Add the top `paper.top_n` picks of `day` as open mock buys (replacing a rerun of the same day)."""
    paper = load_runtime_config()["paper"]
    picks = top.head(int(paper["top_n"]))
    rows = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(day),
            "rank": picks["rank"].to_numpy(),
            "symbol": picks["symbol"].to_numpy(),
            "score": picks["score"].round(6).to_numpy(),
            "model_trained_to": model_trained_to,
            "decided_at": f"{pd.Timestamp.now(tz=MARKET_TZ):%Y-%m-%d %H:%M:%S}",
            "cutoff_price": picks["close_cutoff"].to_numpy(),
            "status": "open",
        }
    ).reindex(columns=COLUMNS)
    numeric = ["buy_price", "sell_price", "shares", "gross_return", "net_return", "pnl"]
    rows[numeric] = rows[numeric].astype(float)
    rows["sell_date"] = pd.to_datetime(rows["sell_date"])
    data = load_ledger(ledger)
    kept = data[data["trade_date"] != pd.Timestamp(day)]
    data = rows if kept.empty else pd.concat([kept, rows], ignore_index=True)
    save_ledger(data, ledger)
    print(f"Paper {ledger}: mock-bought {', '.join(rows['symbol'])} on {day:%Y-%m-%d} -> {ledger_path(ledger).relative_to(ROOT_DIR)}")
    return rows


def next_session(day: pd.Timestamp) -> pd.Timestamp:
    return [session for session in market_sessions(day, day + pd.Timedelta(days=10)) if session > day][0]


def fill_prices(day: pd.Timestamp, symbols: list[str]) -> pd.Series:
    """VWAP of each symbol's 1m bar starting at the cutoff (NaN until the full-session file exists)."""
    config = minutes_pull.load_config()
    storage = Storage(config["storage"], ROOT_DIR)
    raw_dir = Path(config["version"]["data"])
    files = files_by_day(p for p in storage.list(raw_dir, f"{day:%Y%m%d}*.dbn.zst")).get(day, [])
    if not files:
        return pd.Series(np.nan, index=symbols)
    minutes = MinutesFeatureManager(ROOT_DIR / "config/feature/minutes_feature_config.yaml")
    cutoff = session_bounds(day, minutes.cutoff_before_close)[2]
    bars = load_stored_dbn_bars(storage, files)
    bar = bars[(bars["datetime_utc"] == cutoff) & bars["symbol"].isin(symbols)].set_index("symbol")
    return bar["vwap"].reindex(symbols)


def open_prices(day: pd.Timestamp, symbols: list[str]) -> pd.Series:
    """Regular-session opening trade of `day` (NaN until that session's bars exist)."""
    config = session_bars.load_config()
    storage = Storage(config["storage"], ROOT_DIR)
    path = Path(config["storage"]["output"]["session_bars"]) / config["version"]["data"] / session_filename(day)
    if not storage.exists(path):
        return todays_open_prices(day, symbols)
    bars = load_session_bars(storage, [path])
    return bars.set_index("symbol")["open"].reindex(symbols)


def todays_open_prices(day: pd.Timestamp, symbols: list[str]) -> pd.Series:
    """The 09:30 1m bar's open for `day`, straight from the historical API (same-day access).

    Before the nightly job builds `day`'s session bars, this gives the same price (the
    session open is the 09:30 bar's open). Same-day data on the plan is 15 minutes delayed,
    so the 09:30-09:31 bar is available from about 09:46. NaN before then."""
    open_utc = session_bounds(day, 0)[0]
    if pd.Timestamp.now(tz="UTC") < open_utc + pd.Timedelta(minutes=1 + SAME_DAY_DELAY_MINUTES):
        return pd.Series(np.nan, index=symbols)
    import databento

    config = minutes_pull.load_config()
    renames = load_renames()
    tickers = {preprocess_symbol(ticker_for_date(symbol, day, renames)): symbol for symbol in symbols}
    client = databento.Historical(config["provider"]["api_key"])
    store = client.timeseries.get_range(
        dataset=config["provider"]["dataset"], schema="ohlcv-1m", symbols=sorted(tickers),
        start=open_utc, end=open_utc + pd.Timedelta(minutes=1),
    )
    bars = store.to_df().reset_index()
    if bars.empty:
        return pd.Series(np.nan, index=symbols)
    opens = bars.groupby("symbol")["open"].first()
    opens.index = opens.index.map(lambda ticker: tickers.get(ticker, ticker))
    return opens.reindex(symbols)


def settle(ledger: str) -> pd.DataFrame:
    """Fill buy prices (cutoff-minute VWAP) and sell prices (next open) where the data now exists."""
    paper = load_runtime_config()["paper"]
    cost = float(load_runtime_config()["evaluation"]["cost_per_trade"])
    data = load_ledger(ledger)
    if data.empty:
        print(f"Paper {ledger}: no trades yet")
        return data
    was_closed = data["status"] == "closed"
    todo = data[data["status"] != "closed"]
    for day, rows in todo.groupby("trade_date"):
        index, symbols = rows.index, rows["symbol"].tolist()
        if rows["buy_price"].isna().any():
            data.loc[index, "buy_price"] = fill_prices(day, symbols).round(4).to_numpy()
        sell_day = next_session(day)
        data.loc[index, "sell_date"] = sell_day
        if rows["sell_price"].isna().any():
            data.loc[index, "sell_price"] = open_prices(sell_day, symbols).to_numpy()
    done = data["buy_price"].notna() & data["sell_price"].notna()
    data.loc[done, "shares"] = np.floor(amount_per_stock(paper) / data.loc[done, "buy_price"])
    data.loc[done, "gross_return"] = (data.loc[done, "sell_price"] / data.loc[done, "buy_price"] - 1).round(6)
    data.loc[done, "net_return"] = (data.loc[done, "gross_return"] - cost).round(6)
    data.loc[done, "pnl"] = (
        data.loc[done, "shares"] * (data.loc[done, "sell_price"] - data.loc[done, "buy_price"])
        - cost * data.loc[done, "shares"] * data.loc[done, "buy_price"]
    ).round(2)
    data.loc[done, "status"] = "closed"
    data.loc[~done & data["buy_price"].notna(), "status"] = "bought"
    save_ledger(data, ledger)
    print(f"Paper {ledger}: {int(done.sum())} closed, {int((~done).sum())} waiting for prices")
    newly_closed = data.loc[done & ~was_closed, "sell_date"].dropna().unique()
    if len(newly_closed):
        try:
            from .notify import telegram

            config = telegram.load_config()
            if config.get("results_source") == "ledger" and config.get("ledger") == ledger:
                for sell_day in sorted(newly_closed):
                    telegram.notify_sells(pd.Timestamp(sell_day), config)
            elif ledger in telegram.compared(config):  # a model compared side by side
                for sell_day in sorted(newly_closed):
                    telegram.notify_sells(pd.Timestamp(sell_day), config, ledger=ledger, label=telegram.compared(config)[ledger])
        except Exception as error:  # messages never stop the ledger
            print(f"Telegram: {error}")
    return data


def report(ledger: str) -> None:
    data = load_ledger(ledger)
    if data.empty:
        print(f"No trades in {ledger_path(ledger)}")
        return
    shown = data.assign(
        trade_date=data["trade_date"].dt.strftime("%Y-%m-%d"), sell_date=data["sell_date"].dt.strftime("%Y-%m-%d")
    )[["trade_date", "rank", "symbol", "cutoff_price", "buy_price", "sell_date", "sell_price", "net_return", "pnl", "status"]]
    print(shown.to_string(index=False))
    closed = data[data["status"] == "closed"]
    if closed.empty:
        return
    slippage = (closed["buy_price"] / closed["cutoff_price"] - 1).mean()
    print(f"\nClosed trades, buy price vs cutoff price {slippage:+.3%} on average")
    for size in load_runtime_config()["paper"].get("report_sizes", [int(closed["rank"].max())]):
        picks = closed[closed["rank"] <= size]
        daily = picks.groupby("trade_date")["net_return"].mean()
        print(
            f"  top {size:>2}: {len(daily)} days, mean daily return {daily.mean():+.3%} (days up {(daily > 0).mean():.0%}), "
            f"per trade {picks['net_return'].mean():+.3%}, win rate {(picks['net_return'] > 0).mean():.0%}, "
            f"P&L {picks['pnl'].sum():+,.2f} on ${amount_per_stock(load_runtime_config()['paper']) * size:,.0f}/day"
        )


def simulate(start: pd.Timestamp, end: pd.Timestamp, download: bool) -> None:
    """Replay past sessions through the live steps with a weekly model that only knew earlier labels."""
    from . import live  # live imports this module

    sim_model = ROOT_DIR / load_runtime_config()["paper"]["sim_model_path"]
    days = [day for day in market_sessions(start, end) if not is_early_close(day)]
    week = None
    for day in days:
        if week != day.to_period("W-FRI"):
            week = day.to_period("W-FRI")
            live.retrain(as_of=day, model_path=sim_model)
        if download:
            live.step(f"download minute bars for {day:%Y-%m-%d}", lambda: minutes_pull.main(day, day))
        live.predict(day, skip_pull=True, model_path=sim_model, ledger="sim")
    settle("sim")
    report("sim")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading ledger")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("settle", help="Fill buy/sell prices where the data now exists")
    run.add_argument("--ledger", default="live", help="Ledger name: live, late, sim, live_mini, late_mini")
    show = commands.add_parser("report", help="Print the ledger and a summary")
    show.add_argument("--ledger", default="live", help="Ledger name: live, late, sim, live_mini, late_mini")
    sim = commands.add_parser("simulate", help="Replay past sessions: predict, mock buy, mock sell")
    sim.add_argument("--start", required=True)
    sim.add_argument("--end", required=True)
    sim.add_argument("--download", action="store_true", help="Re-download each day's minute bars first (Databento cost)")
    args = parser.parse_args()
    if args.command == "settle":
        settle(args.ledger)
    elif args.command == "report":
        report(args.ledger)
    else:
        simulate(pd.Timestamp(args.start), pd.Timestamp(args.end), args.download)


if __name__ == "__main__":
    main()
