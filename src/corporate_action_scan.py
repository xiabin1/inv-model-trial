"""Advance detection of splits and spin-offs, before their ex-date (complements the price-gap detectors).

Sources:
* Nasdaq Trader corporate-action alerts ("ECA", Nasdaq-listed stocks): the RSS feed lists new
  alerts; each alert page states the ex-date and the terms (split ratio, or spin-off child and
  ratio). New ones for our universe become pending entries with the terms filled in
  (src/corporate_action_flow.py `propose_known`) and a Telegram message with Verify / Reject.
* SEC EDGAR full-text search (all listed companies): 8-K / Form 10 filings mentioning a stock
  split, spin-off or distribution ratio. Filings rarely state a clean ex-date, so these only go
  into a Telegram heads-up digest. Needs SEC_USER_AGENT in .env (SEC asks automated clients to
  identify themselves with a contact); skipped otherwise.

    python -m src.corporate_action_scan scan                 # both sources (cron, twice a day)
    python -m src.corporate_action_scan audit --start 2024-07-01   # EDGAR look-back report

The OCC information-memo site blocks automated access (bot check), so it is not used.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

from .corporate_action_flow import STATE_DIR, load_state, propose_known, read_entries, save_state

ROOT_DIR = Path(__file__).resolve().parents[1]
SEEN_PATH = STATE_DIR / "corporate_action_scan.json"
NASDAQ_RSS = "https://www.nasdaqtrader.com/rss.aspx?feed=currentheadlines"
NASDAQ_ALERT = "https://www.nasdaqtrader.com/TraderNews.aspx?id={id}"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_TICKERS = "https://www.sec.gov/files/company_tickers.json"
EDGAR_QUERIES = ['"stock split"', '"reverse stock split"', '"spin-off"', '"distribution ratio"']
EDGAR_FORMS = "8-K,10-12B,10-12B/A"
BROWSER = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) portfolio-calc-data"}
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
           "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
           "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}


def universe() -> set[str]:
    """Symbols of the latest point-in-time snapshot (the stocks we can pick)."""
    from .symbols.symbol_manager import SymbolManager

    config = SymbolManager.load_config(str(ROOT_DIR / "config/symbols/symbols_config.yaml"))
    _, snapshot = SymbolManager(config["selection"]["random_state"]).load_latest_symbol_snapshot(
        config, pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize(), version="v1")
    return set(snapshot["symbol"].str.upper())


def number(text: str) -> float | None:
    """'25', '0.5', 'one', 'twenty five', 'twenty-five' -> number."""
    text = text.strip().lower().replace("-", " ")
    try:
        return float(text)
    except ValueError:
        words = [NUMBERS.get(word) for word in text.split()]
        return float(sum(words)) if words and all(w is not None for w in words) else None


def page_text(url: str) -> str:
    raw = requests.get(url, headers=BROWSER, timeout=30).text
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


# ---------------------------------------------------------------- Nasdaq alerts

def nasdaq_alerts() -> list[dict]:
    """ECA alerts in the RSS feed: [{id, title, kind, tickers}] for splits and spin-offs."""
    feed = requests.get(NASDAQ_RSS, headers=BROWSER, timeout=30).text
    alerts = []
    for item in re.findall(r"<item>(.*?)</item>", feed, re.S):
        title = html.unescape(re.search(r"<title>(.*?)</title>", item, re.S).group(1)).strip()
        found = re.search(r"\b(ECA\d{4}-\d+)\b", title, re.I)
        if not found:
            continue
        lowered = title.lower()
        kind = ("spinoff" if "spin-off" in lowered or "distribution information" in lowered
                else "reverse_split" if "reverse stock split" in lowered
                else "split" if "stock split" in lowered else None)
        if kind:
            alerts.append({"id": found.group(1).upper(), "title": title, "kind": kind,
                           "tickers": re.findall(r"\(([A-Z][A-Z.]{0,5})\)", title)})
    return alerts


DATE = r"(?:[A-Z][a-z]+day,? )?([A-Z][a-z]+ \d{1,2}, \d{4})"


def parse_alert(text: str, kind: str) -> dict:
    """Ex-date and terms from an ECA alert page; missing pieces are None.

    Spin-offs: "EX Date: January 5, 2026" and "Ratio of Spin-Off/ Distribution: One share of
    ... (VSNT) for every twenty five shares of ... held". Splits: "will effect a one-for-ten (1-10)
    reverse split ... effective on Thursday, October 8, 2026" (or "split-adjusted basis on ...")."""
    result = {"date": None, "terms": None}
    if kind in ("split", "reverse_split"):
        for pattern in (rf"split[- ]adjusted basis(?: at the (?:market )?open)?(?: on)? {DATE}",
                        rf"effective (?:at the (?:market )?open )?(?:on )?{DATE}", rf"EX[- ]?Date:\s*{DATE}"):
            found = re.search(pattern, text, re.I)
            if found:
                result["date"] = pd.Timestamp(found.group(1))
                break
        # first "<n>-for-<m>" whose both sides are numbers ("Request for Information" etc. are skipped)
        for ratio in re.finditer(r"\b([A-Za-z]+|\d+(?:\.\d+)?)[- ]for[- ]([A-Za-z]+|\d+(?:\.\d+)?)\b", text):
            new_shares, old_shares = number(ratio.group(1)), number(ratio.group(2))
            if new_shares and old_shares:
                result["terms"] = {"split": new_shares / old_shares}
                break
        return result
    found = re.search(rf"EX[- ]?Date:\s*{DATE}", text, re.I)
    result["date"] = pd.Timestamp(found.group(1)) if found else None
    block = re.search(r"Ratio of Spin-Off/ ?Distribution:\s*(.{0,400}?)\bheld\b", text, re.I)
    if block:
        child = re.search(r"\(([A-Z][A-Z.]{0,5})\)", block.group(1))
        amounts = re.search(r"^(\S+(?: \S+)?) shares? of .*? for every (.+?) shares? of", block.group(1), re.I)
        if child and amounts:
            given, held = number(amounts.group(1)), number(amounts.group(2))
            if given and held:
                result["terms"] = {"spinoff": {"child": child.group(1), "ratio": round(given / held, 6)}}
    return result


def scan_nasdaq(symbols: set[str], seen: dict, notify: bool) -> list[str]:
    done = set(seen.setdefault("eca", []))
    recorded = {(str(e["symbol"]).upper(), pd.Timestamp(e["date"])) for e in read_entries()}
    proposed = []
    for alert in nasdaq_alerts():
        if alert["id"] in done:
            continue
        ours = [t for t in alert["tickers"] if t.replace(".", "-") in symbols]
        if not ours:
            done.add(alert["id"])  # not one of our stocks
            continue
        symbol = ours[0].replace(".", "-")
        details = parse_alert(page_text(NASDAQ_ALERT.format(id=alert["id"])), alert["kind"])
        if details["date"] is None or details["terms"] is None:
            print(f"{alert['id']} {symbol}: ex-date or terms not stated yet; will look again")  # often updated later
            continue
        done.add(alert["id"])
        if (symbol, details["date"]) in recorded:
            continue
        link = NASDAQ_ALERT.format(id=alert["id"])
        if propose_known(symbol, details["date"], details["terms"], f"{alert['id']}: {alert['title'][:120]}",
                         f"Nasdaq {alert['id']}", link, notify=notify):
            proposed.append(f"{symbol} {details['date']:%Y-%m-%d}")
    seen["eca"] = sorted(done)
    return proposed


# ---------------------------------------------------------------- SEC EDGAR

def sec_headers() -> dict | None:
    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    return {"User-Agent": agent} if agent else None


def ticker_ciks(headers: dict) -> dict[int, str]:
    data = requests.get(EDGAR_TICKERS, headers=headers, timeout=30).json()
    return {int(row["cik_str"]): row["ticker"].upper().replace(".", "-") for row in data.values()}


def edgar_hits(headers: dict, start: pd.Timestamp, end: pd.Timestamp, max_pages: int = 3) -> list[dict]:
    """Filings in [start, end] matching the split/spin-off queries: [{id, cik, form, date, query, link, name}]."""
    hits = []
    for query in EDGAR_QUERIES:
        for page in range(max_pages):
            response = requests.get(EDGAR_SEARCH, headers=headers, timeout=30, params={
                "q": query, "forms": EDGAR_FORMS, "dateRange": "custom", "from": page * 100,
                "startdt": f"{start:%Y-%m-%d}", "enddt": f"{end:%Y-%m-%d}"})
            found = response.json().get("hits", {}).get("hits", [])
            for hit in found:
                source, (accession, filename) = hit["_source"], hit["_id"].split(":", 1)
                for cik in source.get("ciks", []):
                    hits.append({"id": hit["_id"], "cik": int(cik), "form": source.get("form"), "date": source.get("file_date"),
                                 "query": query.strip('"'), "name": (source.get("display_names") or ["?"])[0],
                                 "link": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{filename}"})
            if len(found) < 100:
                break
    return hits


def scan_edgar(symbols: set[str], seen: dict, notify: bool, days: int = 4) -> list[str]:
    headers = sec_headers()
    if headers is None:
        print("EDGAR skipped: set SEC_USER_AGENT in .env (e.g. 'portfolio-calc-data your@contact')")
        return []
    ciks = ticker_ciks(headers)
    end = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    done = set(seen.setdefault("edgar", []))
    lines = []
    for hit in edgar_hits(headers, end - pd.Timedelta(days=days), end):
        symbol = ciks.get(hit["cik"])
        if symbol not in symbols or f"{hit['id']}|{symbol}" in done:
            continue
        done.add(f"{hit['id']}|{symbol}")
        lines.append(f"• <b>{symbol}</b> {hit['form']} {hit['date']} mentions “{hit['query']}” — <a href=\"{hit['link']}\">filing</a>")
    seen["edgar"] = sorted(done)[-5000:]
    if lines and notify:
        from .notify import telegram

        telegram.send("🔎 <b>Possible upcoming splits / spin-offs (SEC filings)</b>\n" + "\n".join(sorted(set(lines)))
                      + "\nHeads-up only: if one is a real event, add it to config/symbols/corporate_actions.yaml "
                        "(verified: false) or wait for the exchange alert.")
    return lines


def scan(notify: bool = True) -> None:
    seen = load_state(SEEN_PATH)
    symbols = universe()
    proposed = scan_nasdaq(symbols, seen, notify)
    headsup = scan_edgar(symbols, seen, notify)
    save_state(SEEN_PATH, seen)
    print(f"Scan: {len(proposed)} pending entr(ies) from Nasdaq alerts {proposed or ''}; {len(headsup)} EDGAR heads-up filing(s)")


def edgar_hits_by_company(headers: dict, ciks: dict[int, str], symbols: set[str], start: pd.Timestamp,
                          end: pd.Timestamp, queries: list[str] = EDGAR_QUERIES) -> list[dict]:
    """Like `edgar_hits`, one company at a time (8-K filings of that company), so no result cap
    applies across companies. About 4 requests per company, paced under the SEC's 10/second."""
    import time

    hits = []
    targets = sorted((cik, symbol) for cik, symbol in ciks.items() if symbol in symbols)
    for number, (cik, symbol) in enumerate(targets, 1):
        for query in queries:
            for page in range(3):
                time.sleep(0.15)
                found = requests.get(EDGAR_SEARCH, headers=headers, timeout=30, params={
                    "q": query, "ciks": f"{cik:010d}", "forms": "8-K", "dateRange": "custom", "from": page * 100,
                    "startdt": f"{start:%Y-%m-%d}", "enddt": f"{end:%Y-%m-%d}"}).json().get("hits", {}).get("hits", [])
                for hit in found:
                    source, (accession, filename) = hit["_source"], hit["_id"].split(":", 1)
                    hits.append({"id": hit["_id"], "cik": cik, "form": source.get("form"), "date": source.get("file_date"),
                                 "query": query.strip('"'), "name": (source.get("display_names") or ["?"])[0],
                                 "link": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filename}"})
                if len(found) < 100:
                    break
        if number % 50 == 0:
            print(f"  {number}/{len(targets)} companies searched", flush=True)
    return hits


# Spin-offs worth only part of the parent are invisible to the split / +-40% price rules. A down-gap
# of 10% or more in a company that mentioned a spin-off in an 8-K during the last 120 days is the
# combination that found APTV, BDX and FTV in the 2026-10-07 audit. Tested on 2024-07..2026-10:
# about 1 alert a month, and every spin-off with a gap of 10% or more was caught (APTV, BDX, CTVA, DD,
# FDX, FTV); smaller ones (HON -5%) still need the Nasdaq alerts.
SPINOFF_GAP = 0.10
SPINOFF_FILING_DAYS = 120
SPINOFF_QUERIES = ['"spin-off"', '"distribution ratio"']


def spinoff_gap_check(day: pd.Timestamp, gaps: pd.Series, skip: set[str] = frozenset(),
                      min_drop: float = SPINOFF_GAP, filing_days: int = SPINOFF_FILING_DAYS) -> list[str]:
    """Propose a pending entry for each stock whose overnight gap into `day` (open / previous close,
    after the recorded actions) is a drop of `min_drop` or more and whose company filed an 8-K
    mentioning a spin-off or distribution ratio in the previous `filing_days` days. Returns the
    proposed symbols. The stock is then excluded from the picks until it is verified or rejected."""
    from .corporate_action_flow import propose

    day = pd.Timestamp(day).normalize()
    drops = gaps[(gaps <= 1 - min_drop) & ~gaps.index.isin(list(skip))]
    if drops.empty:
        return []
    headers = sec_headers()
    if headers is None:
        print(f"Spin-off check skipped for {sorted(drops.index)}: set SEC_USER_AGENT in .env")
        return []
    ciks = ticker_ciks(headers)
    hits = edgar_hits_by_company(headers, ciks, set(drops.index), day - pd.Timedelta(days=filing_days), day,
                                 queries=SPINOFF_QUERIES)
    latest: dict[str, dict] = {}
    for hit in sorted(hits, key=lambda h: h["date"]):
        latest[ciks[hit["cik"]]] = hit
    proposed = []
    for symbol, hit in sorted(latest.items()):
        found_by = (f"nightly check: {drops[symbol] - 1:+.1%} gap and an SEC 8-K mentioning a {hit['query']} "
                    f"({hit['date']}): {hit['link']}")
        if propose(symbol, day, float(drops[symbol]), found_by):
            proposed.append(symbol)
    print(f"Spin-off check {day:%Y-%m-%d}: {len(drops)} drop(s) of {min_drop:.0%}+, "
          f"{len(latest)} with a spin-off filing, proposed {proposed or 'none'}")
    return proposed


def audit(start: pd.Timestamp, notify: bool = True, per_company: bool = False) -> pd.DataFrame:
    """EDGAR filings since `start` for our stocks, minus the events already recorded (within 60 days)."""
    headers = sec_headers()
    if headers is None:
        raise SystemExit("Set SEC_USER_AGENT in .env first")
    ciks, symbols = ticker_ciks(headers), universe()
    end = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    hits = pd.DataFrame(edgar_hits_by_company(headers, ciks, symbols, pd.Timestamp(start), end) if per_company
                        else edgar_hits(headers, pd.Timestamp(start), end, max_pages=10))
    if hits.empty:
        print("No filings found"); return hits
    hits["symbol"] = hits["cik"].map(ciks)
    hits = hits[hits["symbol"].isin(symbols)].copy()
    hits["date"] = pd.to_datetime(hits["date"])
    recorded = pd.DataFrame([{"symbol": str(e["symbol"]).upper(), "event": pd.Timestamp(e["date"])} for e in read_entries()])
    def covered(row) -> bool:
        events = recorded.loc[recorded["symbol"] == row["symbol"], "event"]
        return bool(((events - row["date"]).abs() <= pd.Timedelta(days=60)).any())
    report = hits[~hits.apply(covered, axis=1)].sort_values(["symbol", "date"])
    summary = report.groupby("symbol").agg(filings=("id", "nunique"), first=("date", "min"), last=("date", "max"),
                                           queries=("query", lambda q: ", ".join(sorted(set(q)))))
    path = ROOT_DIR / "logs/corporate_action_audit.csv"
    report.to_csv(path, index=False)
    print(summary.to_string() if not summary.empty else "Every filing matches a recorded event")
    print(f"Details: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Advance detection of splits and spin-offs")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("scan", help="Nasdaq alerts + EDGAR heads-up (cron)")
    run.add_argument("--no-telegram", action="store_true")
    look = commands.add_parser("audit", help="EDGAR look-back: filings not matched by a recorded event")
    look.add_argument("--start", default="2024-07-01")
    look.add_argument("--per-company", action="store_true", help="Search each company separately (no result cap; ~4 requests per company)")
    args = parser.parse_args()
    if args.command == "scan":
        scan(notify=not args.no_telegram)
    else:
        audit(pd.Timestamp(args.start), per_company=args.per_company)


if __name__ == "__main__":
    main()
