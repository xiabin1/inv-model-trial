"""Check-and-approve workflow for splits and spin-offs found by the detectors.

    detector (15:58 gap guard / nightly check)
      -> propose(): pending entry in config/symbols/corporate_actions.yaml (verified: false)
         + Telegram message with Verify / Reject buttons; the stock is excluded from the picks
    you check the announcement and tap a button (or edit the file: verified: true / rejected: true)
      -> check() (every 15 minutes in trading hours, and after the nightly run):
         reads the taps, fills spin-off opening prices, rebuilds the affected days in every data
         version, retrains the live models, confirms in Telegram; the stock is picked again.

    python -m src.corporate_action_flow pending                 # list pending entries
    python -m src.corporate_action_flow check                   # process taps + apply (cron)
    python -m src.corporate_action_flow remind                  # Telegram list of pending entries (cron)
    python -m src.corporate_action_flow propose CTVA 2026-10-01 --gap 0.186 [--split 5]
    python -m src.corporate_action_flow apply CTVA 2026-10-01   # rebuild one event by hand

Pending entries are never applied to prices (src/data/raw/corporate_actions.py `load_actions`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import yaml

from .data.raw.corporate_actions import ACTIONS_PATH, load_actions, nearest_split

ROOT_DIR = Path(__file__).resolve().parents[1]
MARKET_TZ = "America/New_York"
STATE_DIR = ROOT_DIR / "data/state"
APPLIED_PATH = STATE_DIR / "applied_actions.json"
UPDATES_PATH = STATE_DIR / "telegram_updates.json"
LOCK_PATH = STATE_DIR / "corporate_action_apply.lock"
MARKER = "  # Proposed by the detectors (src/corporate_action_flow.py); appended below this line"
# Data versions rebuilt for an event, and the live models retrained afterwards.
MINUTE_CONFIGS = {"": "2025-06-01", "_mini": "2024-08-01", "_b1542": "2024-08-01", "_b1558": "2024-08-01", "_l250": "2024-08-01", "_mini_l250": "2024-08-01"}
COMBINED_STARTS = {"": "2025-06-01", "_mini": "2024-10-01", "_b1542": "2024-10-01", "_b1558": "2024-10-01", "_l250": "2024-10-01", "_mini_l250": "2024-10-01"}
LABEL_CONFIGS = {"": "2025-06-01", "_mini": "2024-08-01", "_l250": "2024-08-01"}
DAILY_CONFIGS = ("", "_l250")
LIVE_PROFILES = ("basic", "mini", "b1542", "b1542_l250", "mini_l250")


# ---------------------------------------------------------------- entries

def read_entries(path: Path | None = None) -> list[dict]:
    path = Path(path or ACTIONS_PATH)
    with Path(path).open(encoding="utf-8") as file:
        return (yaml.safe_load(file) or {}).get("actions") or []


def key(entry: dict) -> str:
    return f"{str(entry['symbol']).upper()} {pd.Timestamp(entry['date']):%Y-%m-%d}"


def status(entry: dict) -> str:
    if entry.get("rejected"):
        return "rejected"
    return "verified" if entry.get("verified") is True else "pending"


def complete(entry: dict) -> bool:
    """Has what is needed to apply it (a spin-off's opening prices are filled at apply time)."""
    spinoff = entry.get("spinoff") or {}
    return "split" in entry or "factor" in entry or {"child", "ratio"} <= set(spinoff)


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def applied_state() -> dict:
    """Entries already built into the data. First run: everything verified today counts as applied."""
    if not APPLIED_PATH.exists():
        state = {key(e): {"status": "verified", "applied_at": "before workflow"} for e in read_entries() if status(e) == "verified"}
        save_state(APPLIED_PATH, state)
    return load_state(APPLIED_PATH)


def excluded_symbols(day: pd.Timestamp) -> dict[str, str]:
    """Stocks to leave out of `day`'s picks: pending entries, and verified ones not yet rebuilt."""
    done = applied_state()
    excluded = {}
    for entry in read_entries():
        if pd.Timestamp(entry["date"]) > pd.Timestamp(day):
            continue
        state = status(entry)
        if state == "pending":
            excluded[str(entry["symbol"]).upper()] = f"pending verification ({key(entry)})"
        elif state == "verified" and key(entry) not in done:
            excluded[str(entry["symbol"]).upper()] = f"verified, waiting for the rebuild ({key(entry)})"
    return excluded


def entry_line(symbol: str, day: pd.Timestamp, gap_ratio: float, found_by: str, split: float | None) -> str:
    found = f"found: {pd.Timestamp.now(tz=MARKET_TZ):%Y-%m-%d}, found_by: \"{found_by}\""
    if split:
        return (f"  - {{symbol: {symbol}, date: {day:%Y-%m-%d}, split: {split:g}, verified: false, {found}, "
                f"source: \"proposed: open/prev close {gap_ratio:.4f} looks like a {split:g}-for-1 split; check the announcement\"}}")
    return (f"  - {{symbol: {symbol}, date: {day:%Y-%m-%d}, type: unknown, verified: false, {found}, "
            f"source: \"proposed: unexplained open/prev close {gap_ratio:.4f}; spin-off? add spinoff: {{child: TICKER, ratio: N}} "
            f"then verify, or reject if it is a real price move\"}}")


def propose(symbol: str, day: pd.Timestamp, gap_ratio: float, found_by: str, path: Path | None = None, notify: bool = True) -> bool:
    """Add a pending entry (once per symbol and date) and announce it. Returns False if it exists."""
    path = Path(path or ACTIONS_PATH)
    symbol, day = symbol.upper(), pd.Timestamp(day).normalize()
    if any(key(e) == f"{symbol} {day:%Y-%m-%d}" for e in read_entries(path)):
        return False
    split = nearest_split(gap_ratio)
    text = Path(path).read_text(encoding="utf-8")
    if MARKER not in text:
        raise ValueError(f"{path}: marker line for proposals not found")
    text = text.replace(MARKER, MARKER + "\n" + entry_line(symbol, day, gap_ratio, found_by, split), 1)
    Path(path).write_text(text, encoding="utf-8")
    load_actions.cache_clear()
    print(f"Proposed corporate action {symbol} {day:%Y-%m-%d} (gap ratio {gap_ratio:.4f}, split guess {split}) -> pending")
    if notify:
        announce(symbol, day, gap_ratio, found_by, split)
    return True


def propose_known(symbol: str, day: pd.Timestamp, terms: dict, source: str, found_by: str, link: str = "",
                  path: Path | None = None, notify: bool = True) -> bool:
    """Pending entry from an announcement with known terms: {"split": 4} or
    {"spinoff": {"child": "VSNT", "ratio": 0.04}}. Once per symbol and date; returns False if it exists."""
    path = Path(path or ACTIONS_PATH)
    symbol, day = symbol.upper(), pd.Timestamp(day).normalize()
    if any(key(e) == f"{symbol} {day:%Y-%m-%d}" for e in read_entries(path)):
        return False
    if "split" in terms:
        detail = f"split: {float(terms['split']):g}"
    else:
        spin = terms["spinoff"]
        detail = f"spinoff: {{child: {spin['child'].upper()}, ratio: {float(spin['ratio']):g}}}"
    found = f"found: {pd.Timestamp.now(tz=MARKET_TZ):%Y-%m-%d}, found_by: \"{found_by}\""
    line = f"  - {{symbol: {symbol}, date: {day:%Y-%m-%d}, {detail}, verified: false, {found}, source: \"{source.replace(chr(34), chr(39))}\"}}"
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        raise ValueError(f"{path}: marker line for proposals not found")
    path.write_text(text.replace(MARKER, MARKER + "\n" + line, 1), encoding="utf-8")
    load_actions.cache_clear()
    print(f"Proposed corporate action {symbol} {day:%Y-%m-%d} ({detail}) from {found_by} -> pending")
    if notify:
        from .notify import telegram

        telegram.send(
            f"📅 <b>Upcoming corporate action: {symbol}</b>, ex-date {day:%a %Y-%m-%d}\n"
            f"{detail.replace('{', '(').replace('}', ')')}\nsource: {found_by}{f' — {link}' if link else ''}\n"
            f"{source}\n<b>{symbol} will be excluded from the picks from {day:%m-%d} until you decide.</b>\n"
            "Check the announcement; Verify applies it on the ex-date (spin-off prices are filled from the ex-date open).",
            buttons=[[{"text": "✅ Verify", "callback_data": f"ca|verify|{symbol}|{day:%Y-%m-%d}"},
                      {"text": "❌ Reject", "callback_data": f"ca|reject|{symbol}|{day:%Y-%m-%d}"}]])
    return True


def set_flag(symbol: str, day: pd.Timestamp, decision: str, path: Path | None = None) -> bool:
    """decision 'verify' -> verified: true; 'reject' -> rejected: true (on the entry's line)."""
    path = Path(path or ACTIONS_PATH)
    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(rf"(?m)^(  - \{{symbol: {re.escape(symbol.upper())}, date: {pd.Timestamp(day):%Y-%m-%d},.*)$")
    match = pattern.search(text)
    if not match:
        return False
    line = match.group(1)
    if decision == "verify":
        new = line.replace("verified: false", "verified: true")
    else:
        new = line if "rejected: true" in line else line.replace("verified: false", "verified: false, rejected: true")
    Path(path).write_text(text.replace(line, new, 1), encoding="utf-8")
    load_actions.cache_clear()
    return new != line


# ---------------------------------------------------------------- Telegram

def announce(symbol: str, day: pd.Timestamp, gap_ratio: float, found_by: str, split: float | None) -> None:
    from .notify import telegram

    guess = (f"looks like a <b>{split:g}-for-1 split</b>" if split and split >= 1
             else f"looks like a <b>1-for-{1 / split:g} reverse split</b>" if split else
             "<b>unexplained</b>: a spin-off, or a real price move")
    text = (f"⚠️ <b>Corporate action to verify: {symbol}</b> ({day:%a %Y-%m-%d})\n"
            f"open / previous close = {gap_ratio:.4f} ({gap_ratio - 1:+.1%}), {guess}\n"
            f"found by: {found_by}\n"
            f"<b>{symbol} is excluded from the picks until you decide.</b>\n")
    buttons = [[{"text": "✅ Verify", "callback_data": f"ca|verify|{symbol}|{day:%Y-%m-%d}"},
                {"text": "❌ Reject", "callback_data": f"ca|reject|{symbol}|{day:%Y-%m-%d}"}]]
    if not split:
        text += ("If it is a spin-off: add <code>spinoff: {child: TICKER, ratio: N}</code> to the entry in "
                 "config/symbols/corporate_actions.yaml (opening prices are filled automatically), then tap Verify.\n"
                 "If it is a real price move: tap Reject.")
    else:
        text += "Check the company's announcement: Verify applies the split; Reject if it is not a split."
    telegram.send(text, buttons=buttons)


def process_telegram(path: Path | None = None) -> list[tuple[str, str]]:
    """Apply Verify / Reject taps from the configured chat; returns [(entry key, decision)]."""
    path = Path(path or ACTIONS_PATH)
    from .notify import telegram

    config = telegram.load_config()
    token, chat = os.environ.get(config["token_env"], ""), str(os.environ.get(config["chat_id_env"], ""))
    if not token or not chat:
        return []
    state = load_state(UPDATES_PATH)
    api = telegram.API.format(token=token, method="{}")
    updates = telegram.requests.get(api.format("getUpdates"), params={"offset": state.get("offset", 0), "timeout": 0,
                                                                      "allowed_updates": json.dumps(["callback_query"])}, timeout=15).json()
    decisions = []
    for update in updates.get("result", []):
        state["offset"] = update["update_id"] + 1
        query = update.get("callback_query")
        if not query or not str(query.get("data", "")).startswith("ca|"):
            continue
        from_chat = str(query.get("message", {}).get("chat", {}).get("id", query.get("from", {}).get("id", "")))
        if from_chat != chat:  # only your chat may approve
            telegram.requests.post(api.format("answerCallbackQuery"), json={"callback_query_id": query["id"], "text": "Not allowed"}, timeout=10)
            continue
        _, decision, symbol, day = query["data"].split("|")
        entry = next((e for e in read_entries(path) if key(e) == f"{symbol} {day}"), None)
        if decision == "verify" and entry is not None and not complete(entry):
            reply = "add the spin-off details first"
            telegram.send(f"⚠️ {symbol} {day}: cannot verify yet: add <code>spinoff: {{child: TICKER, ratio: N}}</code> "
                          "(or a split) to the entry in config/symbols/corporate_actions.yaml, then tap Verify again.")
        else:
            changed = set_flag(symbol, pd.Timestamp(day), decision, path)
            reply = f"{symbol} {day}: {decision}" + ("" if changed else " (already set)")
            decisions.append((f"{symbol} {day}", reply))
            if changed:
                telegram.send(f"📝 Recorded: <b>{symbol} {day}</b> {'verified' if decision == 'verify' else 'rejected'}. "
                              f"{'Applied (rebuild + retrain)' if decision == 'verify' else 'Recorded'} at the next check "
                              f"({next_check():%a %H:%M} ET); you get a confirmation then.")
        telegram.requests.post(api.format("answerCallbackQuery"), json={"callback_query_id": query["id"], "text": reply}, timeout=10)
    save_state(UPDATES_PATH, state)
    return decisions


# ---------------------------------------------------------------- applying

def fill_spinoff_opens(path: Path | None = None) -> None:
    """Fill parent_open / child_open of verified spin-offs from the ex-date's first regular trades."""
    path = Path(path or ACTIONS_PATH)
    from .data.raw.session_bars import load_session_bars, session_daily_bars, session_filename
    from .data.raw.util.normalize import normalize_databento_bars
    from .data.storage import Storage
    from .pull import session_bars

    text = Path(path).read_text(encoding="utf-8")
    for entry in read_entries(path):
        spinoff = entry.get("spinoff") or {}
        if status(entry) != "verified" or not spinoff or {"parent_open", "child_open"} <= set(spinoff):
            continue
        last_session = latest_session_with_data()
        if last_session is None or pd.Timestamp(entry["date"]) > last_session:
            continue  # the ex-date's opening prices are not available yet
        day, parent, child = pd.Timestamp(entry["date"]), str(entry["symbol"]).upper(), str(spinoff["child"]).upper()
        config = session_bars.load_config()
        storage = Storage(config["storage"], ROOT_DIR)
        bars = load_session_bars(storage, [Path(config["storage"]["output"]["session_bars"]) / config["version"]["data"] / session_filename(day)], adjust=False)
        opens = bars.set_index("symbol")["open"]
        if child not in opens.index:
            import databento

            start = pd.Timestamp(f"{day:%Y-%m-%d} 09:30", tz=MARKET_TZ).tz_convert("UTC")
            store = databento.Historical(os.environ["DATABENTO_API_KEY"]).timeseries.get_range(
                dataset="XNAS.BASIC", schema="ohlcv-1m", symbols=[child], start=start, end=start + pd.Timedelta(minutes=30))
            normalized, _ = normalize_databento_bars(store.to_df())
            opens.loc[child] = float(session_daily_bars(normalized).iloc[0]["open"])
        old = re.search(rf"spinoff: \{{child: {child}, ratio: ([\d.]+)\}}", text)
        if old:
            text = text.replace(old.group(0), f"spinoff: {{child: {child}, ratio: {old.group(1)}, parent_open: {float(opens[parent]):g}, "
                                              f"child_open: {float(opens[child]):g}}}")
            print(f"Filled spin-off opens for {parent} {day:%Y-%m-%d}: {float(opens[parent]):g} / {float(opens[child]):g}")
    Path(path).write_text(text, encoding="utf-8")
    load_actions.cache_clear()


def wait_outside_live_window() -> None:
    now = pd.Timestamp.now(tz=MARKET_TZ)
    start, end = now.normalize() + pd.Timedelta(hours=15, minutes=54), now.normalize() + pd.Timedelta(hours=16, minutes=2)
    if start <= now < end:
        print("pausing until 16:02 (live prediction window)", flush=True)
        time.sleep((end - now).total_seconds())


def rebuild(events: list[pd.Timestamp], retrain: bool = True) -> None:
    """Recompute everything an event at ex-date D changes, in every data version:
    labels of D-1; minute features D-1..D+21 sessions; daily features D..D+62; combined D-1..D+62.
    Includes the b1542_l250 data (tag _l250: S&P 500 / Nasdaq-100 plus liquid US stocks)."""
    from . import live
    from .data.features.combined_feature_manager import CombinedFeatureManager
    from .data.features.daily_feature_manager import DailyFeatureManager
    from .data.features.label_manager import LabelManager
    from .data.features.minutes_feature_manager import MinutesFeatureManager
    from .data.raw.util.market_calendar import market_sessions

    today = pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None).normalize()
    sessions = [s for s in market_sessions(pd.Timestamp("2024-07-01"), today) if s < today]

    def spans(after: int, before: int = 0) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        merged = []
        for event in sorted(map(pd.Timestamp, events)):
            i = next((k for k, s in enumerate(sessions) if s >= event), len(sessions) - 1)
            span = (sessions[max(0, i - before)], sessions[min(len(sessions) - 1, i + after)])
            if merged and span[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
            else:
                merged.append(span)
        return merged

    def clip(items, start):
        return [(max(a, pd.Timestamp(start)), b) for a, b in items if b >= pd.Timestamp(start)]

    for tag in DAILY_CONFIGS:
        for a, b in clip(spans(62), "2024-10-01"):
            wait_outside_live_window()
            DailyFeatureManager(ROOT_DIR / f"config/feature/daily_feature_config{tag}.yaml").calculate_features(start_date=a, end_date=b)
    for tag, start in MINUTE_CONFIGS.items():
        for a, b in clip(spans(21, before=1), start):
            wait_outside_live_window()
            MinutesFeatureManager(ROOT_DIR / f"config/feature/minutes_feature_config{tag}.yaml").calculate_features(a, b)
    for tag, start in LABEL_CONFIGS.items():
        for a, _ in clip(spans(0, before=1), start):
            wait_outside_live_window()
            LabelManager(ROOT_DIR / f"config/feature/label_config{tag}.yaml").calculate_labels(a, a)
    for tag, start in COMBINED_STARTS.items():
        for a, b in clip(spans(62, before=1), start):
            wait_outside_live_window()
            CombinedFeatureManager(ROOT_DIR / f"config/feature/combined_feature_config{tag}.yaml").combine(a, b)
    if retrain:
        for profile in LIVE_PROFILES:
            wait_outside_live_window()
            live.retrain(profile=profile)


def latest_session_with_data() -> pd.Timestamp | None:
    """Newest session whose regular-session bars are built (the nightly job builds the previous one)."""
    from .data.raw.dbn_store import file_day
    from .data.storage import Storage
    from .pull import session_bars

    config = session_bars.load_config()
    storage = Storage(config["storage"], ROOT_DIR)
    days = [file_day(p) for p in storage.list(Path(config["storage"]["output"]["session_bars"]) / config["version"]["data"], "*.parquet")]
    days = [d for d in days if not pd.isna(d)]
    return max(days) if days else None


def apply_decisions() -> list[str]:
    """Rebuild for newly verified entries; record newly rejected ones. Returns messages sent."""
    from .notify import telegram

    done = applied_state()
    entries = read_entries()
    rejected = [e for e in entries if status(e) == "rejected" and done.get(key(e), {}).get("status") != "rejected"]
    last_session = latest_session_with_data()
    new = [e for e in entries if status(e) == "verified" and key(e) not in done and complete(e)
           and last_session is not None and pd.Timestamp(e["date"]) <= last_session]  # ex-date data must exist
    messages = []
    for entry in rejected:
        done[key(entry)] = {"status": "rejected", "applied_at": f"{pd.Timestamp.now(tz=MARKET_TZ):%Y-%m-%d %H:%M}"}
        messages.append(f"❌ {key(entry)} rejected: no price adjustment; {entry['symbol']} is included in the picks again.")
    if new:
        fill_spinoff_opens()
        load_actions.cache_clear()
        started = time.time()
        rebuild([pd.Timestamp(e["date"]) for e in new])
        for entry in new:
            action = next((a for a in load_actions() if f"{a.symbol} {a.date:%Y-%m-%d}" == key(entry)), None)
            done[key(entry)] = {"status": "verified", "factor": action.factor if action else None,
                                "applied_at": f"{pd.Timestamp.now(tz=MARKET_TZ):%Y-%m-%d %H:%M}"}
            messages.append(f"✅ {key(entry)} applied (factor {action.factor:.5f}): data rebuilt, models retrained "
                            f"({(time.time() - started) / 60:.0f} min); {entry['symbol']} is included in the picks again.")
    save_state(APPLIED_PATH, done)
    for message in messages:
        print(message)
        telegram.send(message)
    return messages


def next_check(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """When the cron `check` runs next: 07:15 Tue-Sat, every 15 min 09:00-17:45 Mon-Fri (see crontab)."""
    now = now or pd.Timestamp.now(tz=MARKET_TZ)
    t = now.floor("15min") + pd.Timedelta(minutes=15)
    while True:
        minutes = t.hour * 60 + t.minute
        if (t.dayofweek < 5 and 9 * 60 <= minutes <= 17 * 60 + 45 and not 15 * 60 + 54 <= minutes < 16 * 60 + 2) or \
                (1 <= t.dayofweek <= 5 and minutes == 7 * 60 + 15):
            return t
        t += pd.Timedelta(minutes=15)


def ack() -> None:
    """Cron every minute: record Verify / Reject taps right away and answer them (no rebuild)."""
    now = pd.Timestamp.now(tz=MARKET_TZ)
    if now.normalize() + pd.Timedelta(hours=15, minutes=54) <= now < now.normalize() + pd.Timedelta(hours=16, minutes=2):
        return
    if LOCK_PATH.exists():
        return  # the check is running and reads the taps itself
    for entry_key, decision in process_telegram():
        print(f"{now:%Y-%m-%d %H:%M} Telegram: {entry_key} -> {decision}")


def check() -> None:
    """Cron entry point: read Telegram taps, then apply decisions (one run at a time)."""
    now = pd.Timestamp.now(tz=MARKET_TZ)
    if now.normalize() + pd.Timedelta(hours=15, minutes=54) <= now < now.normalize() + pd.Timedelta(hours=16, minutes=2):
        return  # never during the live prediction and buys
    if LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime < 3 * 3600:
        print("another check is still applying; skipped")
        return
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(f"{os.getpid()}")
    try:
        for entry_key, decision in process_telegram():
            print(f"Telegram: {entry_key} -> {decision}")
        apply_decisions()
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def remind() -> None:
    from .notify import telegram

    pending = [e for e in read_entries() if status(e) == "pending"]
    if pending:
        lines = "\n".join(f"• {key(e)}: {e.get('source', '')[:90]}" for e in pending)
        telegram.send(f"⏳ <b>Corporate actions waiting for you ({len(pending)})</b>\n{lines}\n"
                      "These stocks are excluded until verified or rejected (buttons in the earlier messages, or edit the file).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check-and-approve workflow for splits and spin-offs")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pending", help="List pending entries")
    commands.add_parser("check", help="Process Telegram taps and apply decisions (cron)")
    commands.add_parser("ack", help="Record Telegram taps right away, no rebuild (cron, every minute)")
    commands.add_parser("remind", help="Telegram list of pending entries (cron)")
    propose_cmd = commands.add_parser("propose", help="Add a pending entry by hand")
    apply_cmd = commands.add_parser("apply", help="Rebuild the data for one event by hand")
    for command in (propose_cmd, apply_cmd):
        command.add_argument("symbol")
        command.add_argument("date")
    propose_cmd.add_argument("--gap", type=float, required=True, help="Open / previous close on the ex-date, e.g. 0.186")
    args = parser.parse_args()
    if args.command == "pending":
        for entry in read_entries():
            if status(entry) == "pending":
                print(key(entry), "-", entry.get("source", ""))
        print(f"Excluded from picks today: {excluded_symbols(pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None).normalize()) or 'none'}")
    elif args.command == "check":
        check()
    elif args.command == "ack":
        ack()
    elif args.command == "remind":
        remind()
    elif args.command == "propose":
        propose(args.symbol, pd.Timestamp(args.date), args.gap, "by hand")
    else:
        fill_spinoff_opens()
        rebuild([pd.Timestamp(args.date)])
        done = applied_state()
        done[f"{args.symbol.upper()} {pd.Timestamp(args.date):%Y-%m-%d}"] = {"status": "verified", "applied_at": "by hand"}
        save_state(APPLIED_PATH, done)


if __name__ == "__main__":
    main()
