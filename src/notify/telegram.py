"""Telegram messages for the daily trades: buys, sells with P&L, and a P&L summary.

Settings in config/notify/telegram.yaml; the bot token and chat id come from .env
(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). Messages never break trading: failures are printed.

    python -m src.notify.telegram chat-id          # find your chat id (send the bot a message first)
    python -m src.notify.telegram test             # send a test message
    python -m src.notify.telegram summary          # send the P&L summary now

Called from src/broker/ibkr.py (buys; sells after `reconcile`) and src/paper.py `settle`
(sells, when `results_source: ledger`).
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config/notify/telegram.yaml"
API = "https://api.telegram.org/bot{token}/{method}"
TRADE_COLUMNS = ["buy_date", "sell_date", "symbol", "shares", "buy_price", "sell_price", "pnl", "net_return"]


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def send(text: str, config: dict | None = None, buttons: list[list[dict]] | None = None) -> bool:
    """Send an HTML message; returns False (and prints why) instead of raising."""
    try:
        config = config or load_config()
        token, chat_id = os.environ.get(config["token_env"], ""), os.environ.get(config["chat_id_env"], "")
        if not config.get("enabled", True):
            return False
        if not token or not chat_id:
            print(f"Telegram: not sent ({config['token_env']} / {config['chat_id_env']} missing in .env)")
            return False
        response = requests.post(
            API.format(token=token, method="sendMessage"),
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True,
                  **({"reply_markup": {"inline_keyboard": buttons}} if buttons else {})},
            timeout=10,
        )
        if not response.ok:
            print(f"Telegram: send failed {response.status_code}: {response.text[:200]}")
        return response.ok
    except Exception as error:  # a message must never stop trading
        print(f"Telegram: send failed: {error}")
        return False


def money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


# ---------------------------------------------------------------- realized trades

def trades_from_ledger(ledger: str) -> pd.DataFrame:
    from ..paper import load_ledger

    data = load_ledger(ledger)
    data = data[data["status"] == "closed"] if not data.empty else data
    if data.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return data.rename(columns={"trade_date": "buy_date"})[TRADE_COLUMNS].reset_index(drop=True)


def trades_from_ibkr(log: pd.DataFrame) -> pd.DataFrame:
    """Realized round trips from the IBKR log: each filled sell matched to the earliest open buys (FIFO)."""
    if log.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    sent = log[~log["dry_run"].astype(bool) & (log["filled_qty"].fillna(0) > 0)].sort_values(["date", "side"])
    lots: dict[str, list[dict]] = {}
    trades = []
    for _, row in sent.iterrows():
        quantity, price = float(row["filled_qty"]), float(row["avg_fill_price"])
        fee = float(row["commission"]) if pd.notna(row["commission"]) else 0.0
        if row["side"] == "BUY":
            lots.setdefault(row["symbol"], []).append({"date": row["date"], "shares": quantity, "price": price, "fee": fee / quantity})
            continue
        remaining, sell_fee = quantity, fee / quantity
        for lot in lots.get(row["symbol"], []):
            if remaining <= 0 or lot["shares"] <= 0:
                continue
            shares = min(remaining, lot["shares"])
            pnl = shares * (price - lot["price"]) - shares * (lot["fee"] + sell_fee)
            trades.append({"buy_date": lot["date"], "sell_date": row["date"], "symbol": row["symbol"], "shares": shares,
                           "buy_price": lot["price"], "sell_price": price, "pnl": round(pnl, 2),
                           "net_return": pnl / (shares * lot["price"])})
            lot["shares"] -= shares
            remaining -= shares
    return pd.DataFrame(trades, columns=TRADE_COLUMNS)


def realized_trades(config: dict | None = None, ledger: str | None = None) -> pd.DataFrame:
    """Closed trades of the configured source, or of the paper `ledger` when given."""
    config = config or load_config()
    if ledger:
        trades = trades_from_ledger(ledger)
    elif config.get("results_source") == "ibkr":
        from ..broker import ibkr

        trades = trades_from_ibkr(ibkr.load_log(ibkr.load_config()))
    else:
        trades = trades_from_ledger(config["ledger"])
    for column in ("buy_date", "sell_date"):
        trades[column] = pd.to_datetime(trades[column])
    return trades


def price(value: float) -> str:
    """Short price: no decimals from $1,000, one from $100, two below (keeps rows phone-width)."""
    value = float(value)
    return f"{value:.0f}" if value >= 1000 else f"{value:.1f}" if value >= 100 else f"{value:.2f}"


def dollars(value: float) -> str:
    return f"{value:+,.0f}"


def summary_lines(trades: pd.DataFrame, windows: list) -> list[str]:
    """P&L over the last N trading days with sells (by sell date); "all" = every trade so far.

    Rows stay within ~30 characters so they do not wrap on a phone."""
    if trades.empty:
        return ["No closed trades yet."]
    days = sorted(trades["sell_date"].unique())
    lines = [f"{'':<4}{'P&L $':>8}{'ret':>8}{'win':>5}{'up':>5}"]
    periods = [("all", len(days)) if str(n).lower() == "all" else (f"{n}d", int(n)) for n in windows]
    for label, count in periods:
        chosen = trades[trades["sell_date"].isin(days[-count:])]
        invested = (chosen["shares"] * chosen["buy_price"]).sum()
        daily = chosen.groupby("sell_date")["pnl"].sum()
        lines.append(f"{label + ('*' if count > len(days) else ''):<4}{dollars(chosen['pnl'].sum()):>8}"
                     f"{chosen['pnl'].sum() / invested if invested else 0:>+8.2%}{(chosen['pnl'] > 0).mean():>5.0%}"
                     f"{f'{(daily > 0).sum()}/{min(count, len(days))}':>5}")
    if any(str(n).lower() != "all" and int(n) > len(days) for n in windows):
        lines.append(f"* only {len(days)} trading day(s) so far")
    return lines


# ---------------------------------------------------------------- messages

def buy_message(day: pd.Timestamp, orders: pd.DataFrame, dry_run: bool, profile: str, capital: dict | None = None) -> str:
    title = f"{'🧪 DRY RUN — planned buys' if dry_run else '🟢 Bought'} {day:%a %m-%d} ({html.escape(profile)})"
    if capital and capital.get("mode") == "compound":
        title += f"\ncapital ${capital['base']:,.0f} {'+' if capital['pnl'] >= 0 else '-'} P&amp;L ${abs(capital['pnl']):,.0f} = ${capital['capital']:,.0f}"
    elif capital:
        title += f"\ncapital ${capital['capital']:,.0f}"
    lines = [f"{'#':>2} {'sym':<5}{'qty':>5}{'price':>9}{'value':>8}"]
    total = 0.0
    for _, row in orders.sort_values("rank").iterrows():
        filled = row.get("filled_qty")
        if dry_run or pd.isna(filled):
            shares, fill = row["quantity"], row["limit_price"]
        else:
            shares, fill = filled, row.get("avg_fill_price")
        head = f"{int(row['rank']):>2} {row['symbol']:<5}"
        if not dry_run and not float(shares or 0):
            lines.append(f"{head}{'0':>5}{'unfilled':>17}")
            continue
        value = float(shares) * float(fill)
        total += value
        partial = "" if dry_run or float(shares) >= float(row["quantity"]) else f" of {int(row['quantity'])}"
        lines.append(f"{head}{int(shares):>5}{float(fill):>9.2f}{value:>8,.0f}{partial}")
    lines.append(f"{'total':<12}{'':>9}{total:>8,.0f}")
    note = "Limit prices (latest trade +0.3%); nothing was sent." if dry_run else "Average fill prices."
    return f"<b>{title}</b>\n<pre>{html.escape(chr(10).join(lines))}</pre>{note}"


def sell_message(day: pd.Timestamp, sold: pd.DataFrame, summary: list[str], source: str, label: str = "") -> str:
    lines = [f"{'sym':<5}{'qty':>4} {'buy → sell':<11}{'P&L':>6}{'%':>6}"]
    for _, row in sold.sort_values("pnl", ascending=False).iterrows():
        prices = f"{price(row['buy_price'])}→{price(row['sell_price'])}"
        lines.append(f"{row['symbol']:<5}{int(row['shares']):>4} {prices:<11}{dollars(row['pnl']):>6}{row['net_return']:>+6.1%}")
    invested = (sold["shares"] * sold["buy_price"]).sum()
    lines.append(f"{'total':<21}{dollars(sold['pnl'].sum()):>6}{sold['pnl'].sum() / invested if invested else 0:>+6.1%}")
    origin = "IBKR fills" if source == "ibkr" else "paper ledger"
    label = f" ({html.escape(label)})" if label else ""
    return (f"<b>🔴 Sold {day:%a %m-%d}{label}</b> — {(sold['pnl'] > 0).sum()}/{len(sold)} up\n"
            f"<pre>{html.escape(chr(10).join(lines))}</pre>"
            f"<b>P&amp;L</b> ({origin}, after costs)\n<pre>{html.escape(chr(10).join(summary))}</pre>")


def notify_buys(day: pd.Timestamp, orders: pd.DataFrame, dry_run: bool, profile: str, capital: dict | None = None) -> None:
    if not orders.empty:
        send(buy_message(day, orders, dry_run, profile, capital))


def notify_sells(day: pd.Timestamp, config: dict | None = None, ledger: str | None = None, label: str = "") -> bool:
    """Message for the trades sold on `day`, with the P&L summary; nothing if none were sold.
    With `ledger`: the same message for that paper ledger (a model compared side by side)."""
    config = config or load_config()
    trades = realized_trades(config, ledger)
    sold = trades[trades["sell_date"] == pd.Timestamp(day)] if not trades.empty else trades
    if sold.empty:
        return False
    source = "ledger" if ledger else config["results_source"]
    return send(sell_message(pd.Timestamp(day), sold, summary_lines(trades, config["summary_windows"]), source, label), config)


def compared(config: dict | None = None) -> dict[str, str]:
    """Paper ledgers shown next to the main results: {ledger: label} (telegram.yaml `compare`)."""
    config = config or load_config()
    return {item["ledger"]: item.get("label", item["ledger"]) for item in config.get("compare") or []}


def picks_message(day: pd.Timestamp, top: pd.DataFrame, label: str, main: pd.DataFrame | None = None,
                  sectors: dict | None = None) -> str:
    """A compared model's picks at the cutoff (paper only), with how many it shares with the main model."""
    sectors = sectors or {}
    lines = [f"{'#':>2} {'sym':<5}{'price':>9}  sector"]
    for _, row in top.sort_values("rank").iterrows():
        cutoff = row.get("close_cutoff")
        shown = price(cutoff) if pd.notna(cutoff) else "-"
        lines.append(f"{int(row['rank']):>2} {row['symbol']:<5}{shown:>9}  {str(sectors.get(row['symbol'], '?'))[:14]}")
    same = ""
    if main is not None and not main.empty:
        shared = sorted(set(top["symbol"]) & set(main["symbol"]))
        same = f"\n{len(shared)}/{len(top)} same as the traded model"
    return (f"<b>🧪 Picks {day:%a %m-%d} ({html.escape(label)}, paper)</b>\n"
            f"<pre>{html.escape(chr(10).join(lines))}</pre>Mock-bought at the 15:58 minute's average price (XNAS.BASIC).{same}")


def notify_picks(day: pd.Timestamp, top: pd.DataFrame, label: str, main: pd.DataFrame | None = None,
                 sectors: dict | None = None) -> bool:
    return send(picks_message(day, top, label, main, sectors)) if not top.empty else False


def find_chat_id() -> None:
    config = load_config()
    token = os.environ.get(config["token_env"], "")
    if not token:
        print(f"Set {config['token_env']} in .env first")
        return
    updates = requests.get(API.format(token=token, method="getUpdates"), timeout=10).json().get("result", [])
    chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("username") or u["message"]["chat"].get("title")
             for u in updates if "message" in u}
    print(f"Chats that messaged the bot: {chats or 'none - send your bot any message in Telegram, then run this again'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram trade messages")
    parser.add_argument("command", choices=["chat-id", "test", "summary"])
    args = parser.parse_args()
    if args.command == "chat-id":
        find_chat_id()
    elif args.command == "test":
        print("sent" if send("✅ portfolio-calc-data: Telegram messages are working.") else "not sent")
    else:
        config = load_config()
        trades = realized_trades(config)
        print("sent" if send("<b>P&amp;L summary</b>\n<pre>" + html.escape("\n".join(summary_lines(trades, config["summary_windows"]))) + "</pre>") else "not sent")


if __name__ == "__main__":
    main()
