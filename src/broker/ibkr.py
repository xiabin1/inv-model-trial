"""IBKR paper trading of the daily top-N: buy at the cutoff, sell in the next opening auction.

Talks to IB Gateway (logged in to the paper account) through ib_async. Settings are in
config/broker/ibkr_config.yaml: which pipeline's predictions to trade, amounts, the buy
limit buffer, timings, and `dry_run`.

    python -m src.broker.ibkr check              # connect, show account, cash, strategy positions
    python -m src.broker.ibkr buy --date 2026-10-05 [--send]   # top-N limit buys (normally run by `live predict --trade`)
    python -m src.broker.ibkr sell [--send]      # MOO sells of the strategy's open lots (09:00 job)
    python -m src.broker.ibkr reconcile          # record today's fills and commissions in the log

Safety: refuses non-paper accounts (ids not starting with "DU") while `paper_only` is set;
caps each order at `max_order_value`; tags orders with `order_ref` and only ever sells
shares it bought under that tag; never sends the same day's buy or sell twice.
Every order and fill is appended to `log.path`.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import yaml

from ..data.storage import Storage
from ..train import ROOT_DIR, load_runtime_config

MARKET_TZ = "America/New_York"
CONFIG_PATH = ROOT_DIR / "config/broker/ibkr_config.yaml"
LOG_COLUMNS = [
    "date", "side", "symbol", "rank", "ref_price", "quantity", "limit_price", "order_type", "order_id",
    "status", "filled_qty", "avg_fill_price", "commission", "sent_at", "dry_run", "note",
]


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=MARKET_TZ)


def ny_time(day: pd.Timestamp, clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} {clock}").tz_localize(MARKET_TZ)


def ib_symbol(symbol: str) -> str:
    """Pipeline symbol to IBKR's (share classes use a space: BRK-B -> BRK B)."""
    return symbol.replace("-", " ").replace(".", " ")


def daily_capital(day: pd.Timestamp, config: dict, log: pd.DataFrame | None = None) -> dict:
    """Money for `day`'s buys: the starting amount, plus (compound mode) the realized P&L of every
    IBKR trade bought on/after `since` and sold before `day` (fills and commissions from the log)."""
    strategy = config["strategy"]
    if "capital" not in strategy:  # older config: a fixed amount per day
        return {"mode": "fixed", "base": float(strategy["capital_per_day"]), "pnl": 0.0, "capital": float(strategy["capital_per_day"])}
    capital = strategy["capital"]
    base = float(capital["amount"])
    if capital.get("mode", "fixed") != "compound":
        return {"mode": "fixed", "base": base, "pnl": 0.0, "capital": base}
    from ..notify.telegram import trades_from_ibkr

    trades = trades_from_ibkr(load_log(config) if log is None else log)
    if not trades.empty:
        trades = trades[(pd.to_datetime(trades["buy_date"]) >= pd.Timestamp(capital["since"]))
                        & (pd.to_datetime(trades["sell_date"]) < pd.Timestamp(day))]
    pnl = float(trades["pnl"].sum()) if not trades.empty else 0.0
    return {"mode": "compound", "base": base, "pnl": round(pnl, 2), "capital": round(base + pnl, 2),
            "since": f"{pd.Timestamp(capital['since']):%Y-%m-%d}"}


def set_capital(amount: float, since: pd.Timestamp, path: str | Path = CONFIG_PATH) -> None:
    """Set the starting amount and its start date in the config file (comments are kept)."""
    import re

    text = Path(path).read_text(encoding="utf-8")
    text, n_amount = re.subn(r"(?m)^(    amount:\s*)\S+", lambda m: f"{m.group(1)}{amount:.0f}", text)
    text, n_since = re.subn(r"(?m)^(    since:\s*)\S+", lambda m: f"{m.group(1)}{pd.Timestamp(since):%Y-%m-%d}", text)
    if n_amount != 1 or n_since != 1:
        raise ValueError(f"{path}: expected one strategy.capital amount and since line")
    Path(path).write_text(text, encoding="utf-8")
    print(f"Capital set to ${amount:,.0f}, counting P&L of trades bought from {pd.Timestamp(since):%Y-%m-%d}")


def plan_buys(picks: pd.DataFrame, strategy: dict, limit_buffer: float) -> pd.DataFrame:
    """Shares and limit price per pick: equal dollars, whole shares, limit = price x (1 + buffer)."""
    picks = picks.sort_values("rank").head(int(strategy["top_n"])).copy()
    amount = float(strategy["capital_per_day"]) / int(strategy["top_n"])
    picks["ref_price"] = picks["close_cutoff"].astype(float)
    picks["quantity"] = (amount / picks["ref_price"]).apply(math.floor).astype(int)
    picks["limit_price"] = (picks["ref_price"] * (1 + float(limit_buffer))).round(2)
    too_big = picks["quantity"] * picks["limit_price"] > float(strategy["max_order_value"])
    picks.loc[too_big, "quantity"] = (float(strategy["max_order_value"]) / picks.loc[too_big, "limit_price"]).apply(math.floor)
    return picks[picks["quantity"] > 0][["rank", "symbol", "ref_price", "quantity", "limit_price"]]


def live_prices(symbols: list[str], lookback_seconds: int = 120, timeout_seconds: float = 8.0) -> pd.Series:
    """Latest trade price per symbol from the EQUS.MINI live feed (replaying the last few minutes).

    For profiles whose prediction price is stale at order time (b1542 predicts from bars before
    15:42): the limit is set from this price instead. EQUS.MINI live is included in the plan."""
    import os
    import threading

    import databento

    from ..data.raw.util.normalize import preprocess_symbol

    tickers = {preprocess_symbol(symbol): symbol for symbol in symbols}
    names: dict[int, str] = {}
    last: dict[str, float] = {}
    started = pd.Timestamp.now(tz="UTC")
    caught_up = threading.Event()

    def on_record(record) -> None:
        if isinstance(record, databento.SymbolMappingMsg):
            names[record.instrument_id] = record.stype_in_symbol
        elif isinstance(record, databento.TradeMsg):
            symbol = tickers.get(names.get(record.instrument_id, ""))
            if symbol:
                last[symbol] = float(record.pretty_price)
            if record.ts_event >= started.value:
                caught_up.set()  # replay finished: now receiving live trades

    client = databento.Live(key=os.environ["DATABENTO_API_KEY"])
    client.subscribe(dataset="EQUS.MINI", schema="trades", symbols=sorted(tickers), stype_in="raw_symbol",
                     start=started - pd.Timedelta(seconds=lookback_seconds))
    client.add_callback(on_record)
    client.start()
    deadline = pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=timeout_seconds)
    while pd.Timestamp.now(tz="UTC") < deadline and not (caught_up.is_set() and len(last) == len(tickers)):
        caught_up.wait(0.2)
    client.stop()
    client.block_for_close(timeout=5)
    return pd.Series(last, dtype=float).reindex(symbols)


# ---------------------------------------------------------------- log

def log_path(config: dict) -> Path:
    return ROOT_DIR / config["log"]["path"]


def load_log(config: dict) -> pd.DataFrame:
    path = log_path(config)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    data = pd.read_csv(path, dtype={"symbol": str, "status": str, "note": str})
    data["date"] = pd.to_datetime(data["date"])
    return data


def save_log(data: pd.DataFrame, config: dict) -> None:
    path = log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    data[LOG_COLUMNS].sort_values(["date", "side", "symbol"]).to_csv(path, index=False, date_format="%Y-%m-%d")


def append_log(rows: list[dict], config: dict) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows).reindex(columns=LOG_COLUMNS)
    new["date"] = pd.to_datetime(new["date"])
    data = load_log(config)
    save_log(new if data.empty else pd.concat([data, new], ignore_index=True), config)


def open_lots(log: pd.DataFrame) -> pd.Series:
    """Shares the strategy holds: filled buys minus filled sells, per symbol (sent orders only)."""
    sent = log[~log["dry_run"].astype(bool)] if not log.empty else log
    if sent.empty:
        return pd.Series(dtype=float)
    signed = sent["filled_qty"].fillna(0).astype(float).where(sent["side"] == "BUY", -sent["filled_qty"].fillna(0).astype(float))
    lots = signed.groupby(sent["symbol"]).sum()
    return lots[lots > 0]


# ---------------------------------------------------------------- connection

def connect(config: dict):
    """Connected ib_async.IB; refuses non-paper accounts while `paper_only` is set."""
    from ib_async import IB

    connection = config["connection"]
    ib = IB()
    ib.connect(connection["host"], int(connection["port"]), clientId=int(connection["client_id"]),
               timeout=float(connection["timeout_seconds"]))
    accounts = ib.managedAccounts()
    if connection.get("paper_only", True) and not all(account.startswith("DU") for account in accounts):
        ib.disconnect()
        raise RuntimeError(f"Refusing to trade: accounts {accounts} are not all paper accounts (DU...)")
    if connection.get("account") and connection["account"] not in accounts:
        ib.disconnect()
        raise RuntimeError(f"Refusing to trade: the strategy account {connection['account']} is not in {accounts}")
    return ib


def account_of(config: dict) -> str:
    """The strategy's own IBKR account (connection.account), '' if not pinned."""
    return str(config["connection"].get("account") or "")


def qualified(ib, symbols: list[str]) -> dict:
    from ib_async import Stock

    contracts = {symbol: Stock(ib_symbol(symbol), "SMART", "USD") for symbol in symbols}
    ib.qualifyContracts(*contracts.values())
    return {symbol: contract for symbol, contract in contracts.items() if contract.conId}


def commission(fills) -> float:
    return round(sum(fill.commissionReport.commission or 0.0 for fill in fills if fill.commissionReport), 4)


# ---------------------------------------------------------------- actions

def telegram_buys(day: pd.Timestamp, orders: pd.DataFrame, dry_run: bool, profile: str, capital: dict | None = None) -> None:
    try:
        from ..notify import telegram

        telegram.notify_buys(day, orders, dry_run, profile, capital)
    except Exception as error:  # messages never stop trading
        print(f"Telegram: {error}")


def predictions_for(day: pd.Timestamp, profile: str) -> pd.DataFrame:
    """The profile's real-time 15:58 top list for `day` (written by `live predict`)."""
    from ..live import PROFILES

    train = load_runtime_config(PROFILES[profile]["train"])
    storage = Storage(train["storage"], ROOT_DIR)
    path = Path(train["storage"]["output"]["predictions"]) / train["version"]["evaluation"] / f"{day:%Y-%m-%d}.csv"
    if not storage.exists(path):
        raise FileNotFoundError(f"No {profile} predictions for {day:%Y-%m-%d} at {storage.describe(path)}")
    return storage.read_csv(path)


def buy(day: pd.Timestamp, picks: pd.DataFrame | None = None, send: bool | None = None, config: dict | None = None) -> pd.DataFrame:
    """Limit-buy the day's top-N; wait until `cancel_at`, cancel what is unfilled, log the fills."""
    config = config or load_config()
    strategy = config["strategy"]
    send = (not config.get("dry_run", True)) if send is None else send
    picks = predictions_for(day, strategy["profile"]) if picks is None else picks
    buy_config = config["buy"]
    if buy_config.get("wait_until"):
        wait = (ny_time(day, buy_config["wait_until"]) - now()).total_seconds()
        if 0 < wait < 600:
            print(f"IBKR: waiting {wait:.0f}s until {buy_config['wait_until']} to place the buys")
            time.sleep(wait)
    if buy_config.get("price_source", "prediction") == "equs_mini_live":
        prices = live_prices(picks["symbol"].tolist())
        print("IBKR: latest EQUS.MINI trade prices: " + ", ".join(f"{s} {p:.2f}" for s, p in prices.dropna().items()))
        missing = prices[prices.isna()].index.tolist()
        if missing:
            print(f"IBKR: no live price for {missing}; their limit uses the prediction price")
        picks = picks.assign(close_cutoff=prices.reindex(picks["symbol"]).fillna(picks.set_index("symbol")["close_cutoff"]).to_numpy())
    log = load_log(config)
    capital = daily_capital(day, config, log)
    plan = plan_buys(picks, {**strategy, "capital_per_day": capital["capital"]}, buy_config["limit_buffer"])
    done = set(log.loc[(log["date"] == pd.Timestamp(day)) & (log["side"] == "BUY") & ~log["dry_run"].astype(bool), "symbol"]) if not log.empty else set()
    plan = plan[~plan["symbol"].isin(done)]
    print(f"IBKR buy {day:%Y-%m-%d} ({'SEND' if send else 'DRY RUN'}), {strategy['profile']} top {strategy['top_n']}, "
          f"${capital['capital']:,.0f}" + (f" (${capital['base']:,.0f} {capital['pnl']:+,.2f} P&L since {capital['since']})" if capital["mode"] == "compound" else "")
          + f", limit +{float(config['buy']['limit_buffer']):.2%}:")
    print(plan.to_string(index=False) if not plan.empty else "  nothing to buy (already sent today)")
    rows = []
    if plan.empty:
        return plan
    if not send:
        for _, pick in plan.iterrows():
            rows.append({**pick.to_dict(), "date": day, "side": "BUY", "order_type": "LMT", "status": "dry_run",
                         "sent_at": f"{now():%Y-%m-%d %H:%M:%S}", "dry_run": True})
        append_log(rows, config)
        telegram_buys(day, plan, True, strategy["profile"], capital)
        return plan

    from ib_async import LimitOrder

    ib = connect(config)
    try:
        contracts = qualified(ib, plan["symbol"].tolist())
        trades = {}
        for _, pick in plan.iterrows():
            if pick["symbol"] not in contracts:
                rows.append({**pick.to_dict(), "date": day, "side": "BUY", "order_type": "LMT", "status": "not_sent",
                             "sent_at": f"{now():%Y-%m-%d %H:%M:%S}", "dry_run": False, "note": "contract not found"})
                continue
            order = LimitOrder("BUY", int(pick["quantity"]), float(pick["limit_price"]), tif="DAY",
                               orderRef=strategy["order_ref"], outsideRth=False, account=account_of(config))
            trades[pick["symbol"]] = (pick, ib.placeOrder(contracts[pick["symbol"]], order), now())
        deadline = ny_time(day, config["buy"]["cancel_at"])
        while now() < deadline and any(not trade.isDone() for _, trade, _ in trades.values()):
            ib.sleep(1)
        for _, trade, _ in trades.values():
            if not trade.isDone():
                ib.cancelOrder(trade.order)
        ib.sleep(3)
        for symbol, (pick, trade, sent_at) in trades.items():
            filled = sum(fill.execution.shares for fill in trade.fills)
            value = sum(fill.execution.shares * fill.execution.price for fill in trade.fills)
            rows.append({**pick.to_dict(), "date": day, "side": "BUY", "order_type": "LMT", "order_id": trade.order.orderId,
                         "status": trade.orderStatus.status, "filled_qty": filled,
                         "avg_fill_price": round(value / filled, 4) if filled else None,
                         "commission": commission(trade.fills), "sent_at": f"{sent_at:%Y-%m-%d %H:%M:%S}", "dry_run": False})
    finally:
        append_log(rows, config)
        ib.disconnect()
    result = pd.DataFrame(rows)
    print(result[["symbol", "quantity", "limit_price", "status", "filled_qty", "avg_fill_price"]].to_string(index=False))
    telegram_buys(day, result, False, strategy["profile"], capital)
    return result


def sell_targets(lots: pd.Series, held: dict[str, float], sell_all: bool) -> tuple[dict[str, int], dict[str, str]]:
    """Shares to sell at the open per pipeline symbol, and a note where IBKR's position differs.

    With `sell_all` (the account holds only strategy positions), everything IBKR holds is sold: a
    split's extra shares, a spin-off's new shares and leftovers included. Otherwise the strategy's
    lots, capped at IBKR's position."""
    targets, notes = {}, {}
    for symbol, quantity in lots.items():
        position = float(held.get(ib_symbol(symbol), 0))
        targets[symbol] = int(position if sell_all else min(quantity, position))
        if position != quantity:
            notes[symbol] = f"IBKR holds {position:g}, strategy bought {quantity:g}" + (" (all sold)" if sell_all else " (capped)")
    if sell_all:
        bought = {ib_symbol(symbol) for symbol in lots.index}
        for name, position in held.items():
            if name not in bought and position > 0:
                symbol = name.replace(" ", "-")
                targets[symbol] = int(position)
                notes[symbol] = f"IBKR holds {position:g}, not bought by the strategy (split / spin-off / leftover; sold)"
    return targets, notes


def sell(day: pd.Timestamp, send: bool | None = None, config: dict | None = None) -> pd.DataFrame:
    """Market-on-open sells: everything the strategy's account holds (`sell.sell_all_positions`), or the
    strategy's open lots capped at IBKR's position."""
    from ib_async import MarketOrder

    config = config or load_config()
    strategy = config["strategy"]
    send = (not config.get("dry_run", True)) if send is None else send
    log = load_log(config)
    if not log.empty and ((log["date"] == pd.Timestamp(day)) & (log["side"] == "SELL") & ~log["dry_run"].astype(bool)).any():
        print(f"IBKR sell {day:%Y-%m-%d}: already sent today")
        return pd.DataFrame()
    lots = open_lots(log)
    sell_all = bool(config["sell"].get("sell_all_positions", False))
    print(f"IBKR sell {day:%Y-%m-%d} ({'SEND' if send else 'DRY RUN'}, {'all positions' if sell_all else 'strategy lots'}), "
          f"MOO: {lots.to_dict() or 'no open lots'}")
    if not send:
        return lots.rename("quantity").reset_index()
    ib = connect(config)
    rows = []
    try:
        account = account_of(config)
        held = {p.contract.symbol: p.position for p in ib.positions(account) if p.position > 0}
        targets, notes = sell_targets(lots, held, sell_all)
        if not targets:
            print("IBKR sell: nothing held")
            return pd.DataFrame()
        contracts = qualified(ib, list(targets))
        for symbol, quantity in targets.items():
            note = notes.get(symbol, "")
            if quantity <= 0 or symbol not in contracts:
                rows.append({"date": day, "side": "SELL", "symbol": symbol, "quantity": 0, "order_type": "MOO",
                             "status": "not_sent", "dry_run": False, "note": note or "contract not found",
                             "sent_at": f"{now():%Y-%m-%d %H:%M:%S}"})
                continue
            order = MarketOrder("SELL", quantity, tif="OPG", orderRef=strategy["order_ref"], account=account)
            trade = ib.placeOrder(contracts[symbol], order)
            rows.append({"date": day, "side": "SELL", "symbol": symbol, "quantity": quantity, "order_type": "MOO",
                         "order_id": trade.order.orderId, "status": "submitted", "dry_run": False, "note": note,
                         "sent_at": f"{now():%Y-%m-%d %H:%M:%S}"})
        ib.sleep(3)
        if notes:
            try:
                from ..notify import telegram

                telegram.send("⚠️ <b>IBKR position differs from the strategy's buys</b> (sold at the open):\n"
                              + "\n".join(f"• {symbol}: {note}" for symbol, note in notes.items())
                              + "\nCheck for a split or spin-off (corporate actions), or a partial fill.")
            except Exception as error:  # messages never stop trading
                print(f"Telegram: {error}")
    finally:
        append_log(rows, config)
        ib.disconnect()
    return pd.DataFrame(rows)


def reconcile(config: dict | None = None) -> pd.DataFrame:
    """Record today's executions (fills, average price, commission) against the logged orders."""
    from ib_async import ExecutionFilter

    config = config or load_config()
    log = load_log(config)
    ib = connect(config)
    try:
        fills = ib.reqExecutions(ExecutionFilter())
    finally:
        ib.disconnect()
    by_order: dict[int, list] = {}
    for fill in fills:
        if fill.execution.orderRef == config["strategy"]["order_ref"]:
            by_order.setdefault(fill.execution.orderId, []).append(fill)
    for index, row in log.iterrows():
        if pd.isna(row["order_id"]) or int(row["order_id"]) not in by_order:
            continue
        order_fills = by_order[int(row["order_id"])]
        shares = sum(fill.execution.shares for fill in order_fills)
        log.loc[index, "filled_qty"] = shares
        log.loc[index, "avg_fill_price"] = round(sum(f.execution.shares * f.execution.price for f in order_fills) / shares, 4)
        log.loc[index, "commission"] = commission(order_fills)
        log.loc[index, "status"] = "Filled" if shares >= row["quantity"] else "PartiallyFilled"
    save_log(log, config)
    print(f"Reconciled {sum(len(v) for v in by_order.values())} execution(s) for {len(by_order)} order(s)")
    try:
        from ..notify import telegram

        if telegram.load_config().get("results_source") == "ibkr":
            telegram.notify_sells(now().tz_localize(None).normalize())
    except Exception as error:  # messages never stop trading
        print(f"Telegram: {error}")
    return log


def check(config: dict | None = None) -> None:
    config = config or load_config()
    ib = connect(config)
    try:
        accounts = ib.managedAccounts()
        summary = {item.tag: item.value for item in ib.accountSummary() if item.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower")}
        positions = {p.contract.symbol: p.position for p in ib.positions()}
    finally:
        ib.disconnect()
    print(f"Connected to {config['connection']['host']}:{config['connection']['port']}, accounts {accounts} (paper)")
    print(f"Account: {summary}")
    print(f"IBKR positions: {positions or 'none'}")
    print(f"Strategy open lots (from the log): {open_lots(load_log(config)).to_dict() or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR paper trading of the daily top-N")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Connect and show account, cash and positions")
    for name in ("buy", "sell"):
        command = commands.add_parser(name)
        command.add_argument("--date", help="Session (default: today, New York time)")
        command.add_argument("--send", action="store_true", help="Send orders even if dry_run is set in the config")
    commands.add_parser("reconcile", help="Record today's fills in the log")
    money = commands.add_parser("capital", help="Set the starting money (resets the compounding)")
    money.add_argument("amount", nargs="?", type=float, help="Starting amount, e.g. 50000 (omit to show today's capital)")
    money.add_argument("--since", help="Count P&L from this date (default: today)")
    args = parser.parse_args()
    day = pd.Timestamp(args.date) if getattr(args, "date", None) else now().tz_localize(None).normalize()
    if args.command == "check":
        check()
    elif args.command == "buy":
        buy(day, send=True if args.send else None)
    elif args.command == "sell":
        sell(day, send=True if args.send else None)
    elif args.command == "capital":
        if args.amount is not None:
            set_capital(args.amount, pd.Timestamp(args.since) if args.since else day)
        print(f"Capital for {day:%Y-%m-%d}: {daily_capital(day, load_config())}")
    else:
        reconcile()


if __name__ == "__main__":
    main()
