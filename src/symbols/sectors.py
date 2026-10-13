"""Keep config/symbols/sectors.csv (GICS sector per stock, used by the sector cap) up to date.

Weekly (cron): reads the current S&P 500 list with GICS sectors (Wikipedia), adds or updates those
rows and keeps the others (filled by hand). Current index members still without a sector are
reported in Telegram, because the cap does not count them until they are added by hand.

    python -m src.symbols.sectors refresh
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

from .liquidity_universe import INDEX_LIST, latest_list

ROOT_DIR = Path(__file__).resolve().parents[2]
SECTORS_PATH = ROOT_DIR / "config/symbols/sectors.csv"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) portfolio-calc-data"}


def sp500_gics() -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(requests.get(SP500_URL, headers=HEADERS, timeout=30).text))
    table = next(t for t in tables if "Symbol" in t.columns and "GICS Sector" in t.columns)
    return pd.DataFrame({"symbol": table["Symbol"].astype(str).str.replace(".", "-", regex=False),
                         "sector": table["GICS Sector"], "sub_industry": table["GICS Sub-Industry"]})


def refresh(notify: bool = True) -> list[str]:
    """Update the table; returns current index members still missing a sector."""
    current = pd.read_csv(SECTORS_PATH) if SECTORS_PATH.exists() else pd.DataFrame(columns=["symbol", "sector", "sub_industry"])
    gics = sp500_gics()
    merged = pd.concat([current[~current["symbol"].isin(gics["symbol"])], gics]).sort_values("symbol")
    added = sorted(set(gics["symbol"]) - set(current["symbol"]))
    changed = sorted(s for s, sec in zip(gics["symbol"], gics["sector"])
                     if s in set(current["symbol"]) and current.set_index("symbol").at[s, "sector"] != sec)
    merged.to_csv(SECTORS_PATH, index=False)
    members = latest_list(INDEX_LIST, pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize())
    missing = sorted(set(members["symbol"]) - set(merged["symbol"])) if members is not None else []
    print(f"sectors: {len(merged)} stocks; added {added or '-'}; sector changed {changed or '-'}; index members missing {missing or '-'}")
    if missing and notify:
        from ..notify import telegram

        telegram.send("🏷 <b>Sector table</b>: index members without a GICS sector (not counted by the sector cap): "
                      f"{', '.join(missing)}\nAdd them to config/symbols/sectors.csv.")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the GICS sector table")
    parser.add_argument("command", choices=["refresh"])
    parser.parse_args()
    refresh()


if __name__ == "__main__":
    main()
