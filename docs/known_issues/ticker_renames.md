# Renamed tickers missing from historical data

**Status:** fixed 2026-09-28. Old tickers confirmed from company releases / SEC filings
(sources in `config/symbols/corporate_actions.yaml`, `renames:`); P was PSTG, VMRK was EQR
(Equity Residential is the surviving, renamed company), ECHO was SATS. BNY (ticker reuse) is
handled the same way: BNY Mellon traded as BK until 2026-05-20.

How it works now:
- `renames:` in `config/symbols/corporate_actions.yaml` (not `symbols_config.yaml` as planned
  below: all corporate events live in one file).
- `src/data/raw/ticker_renames.py`: `ticker_for_date` for pulls; `apply_renames` in
  `load_dbn_bars` / `load_session_bars` (with `adjust=True`) drops current-ticker bars up to
  `until` and relabels old-ticker bars up to `until`.
- `python -m src.pull.minutes_pull --renames-only --start-date ... --end-date ...` writes
  `YYYYMMDD.fix.ohlcv-1m.dbn.zst` top-up files; range pulls log unresolved symbols.
- Loaders group raw files per day (`files_by_day`), so top-up files are read too. (Before the
  fix, the minutes features and session-bar build kept only one file per day.)
- `raw/daily` (ohlcv-1d) was not topped up: the pipeline uses session bars built from 1m data.

The original plan follows for reference.

## Problem

Symbol snapshots (`symbols/v1/YYYY-MM-DD.csv`) list each company under its **current**
ticker, even for past dates. When a company changed ticker, the new ticker did not
exist on the exchange before the change, so Databento cannot resolve it for earlier
dates. The pulls log:

```
BentoWarning: The streaming request had one or more symbols which did not resolve: ECHO, MRSH, P, ...
```

The request still succeeds; those stocks simply have no bars on those days.

## Affected tickers

Found from the `not_found` list in the headers of the raw files on S3
(`raw/daily/v1`: 434 files 2025-01-02..2026-09-25, `raw/minutes/v1`: 244 files in 2025).

| Ticker (current) | Company | Unresolved through | Daily days missing | Old ticker |
|---|---|---|---|---|
| FISV | Fiserv | 2025-11-10 | 215 | FI |
| PSKY | Paramount Skydance | 2025-08-06 | 148 | PARA |
| MRSH | Marsh McLennan | 2026-01-13 | 258 | MMC |
| P | Everpure | 2026-04-16 | 322 | **unknown, to confirm** |
| ECHO | EchoStar | 2026-06-23 | 368 | SATS (to confirm) |
| VMRK | Vivmark Residential | 2026-08-17 | 406 | **unknown, to confirm** |

All six resolve from 2026-08-18 onward (e.g. `20260925` daily file has no `not_found`).

**Related:** BNY shows the opposite problem, ticker reuse: before about 2026-02 "BNY" was a
different (~$10) security; BNY Mellon uses it from 2026-05-21. See `corporate_actions.md`.

## Impact

- **Size:** 1,717 of about 225,000 daily stock-days (0.76%); minute data in 2025 misses
  4-6 of about 520 stocks every day (1.06%). All six are S&P 500 members.
- **Training:** these large caps are missing for most of the history. The gap is not
  random, but at about 1% of rows the effect on the model should be small.
- **Features right after a rename:** history starts on the rename day. qlib rolling
  operators use `min_periods=1`, so for about 60 trading days windows such as MA60/STD60
  are silently computed from fewer days (values look valid, not NaN). The first day has
  no `ret_gap`, and `rvol` averages fewer days.
- **Live:** VMRK has history only since 2026-08-18, so its 60-day daily features are
  based on partial history until about mid-November 2026.

## Fix

Keep the existing raw files unchanged. Add the missing data as separate top-up files
and rename old tickers to the current ones when loading.

1. **Alias table** in `config/symbols/symbols_config.yaml`: the current ticker, the old
   ticker and the last date the old ticker applies.

   ```yaml
   aliases:
     FISV: {old: FI,   until: 2025-11-10}
     PSKY: {old: PARA, until: 2025-08-06}
     MRSH: {old: MMC,  until: 2026-01-13}
     ECHO: {old: SATS, until: 2026-06-23}   # confirm
     P:    {old: ???,  until: 2026-04-16}   # look up
     VMRK: {old: ???,  until: 2026-08-17}   # look up
   ```

   Confirm every old ticker from a trusted source before use. A wrong entry fills the
   gap with another company's prices, which is worse than leaving it empty.

2. **Helper** (e.g. in `src/symbols/symbol_manager.py`): `ticker_for_date(symbol, day)`
   returns the old ticker when `day <= until`, otherwise the symbol itself.

3. **Pulls request the ticker valid on each date:**
   - `src/pull/daily_pull.py`, `fetch_session` (uses `manager.load_symbol_snapshot`)
   - `src/pull/minutes_pull.py`, `get_puller` (uses `manager.load_latest_symbol_snapshot`)

   Map the snapshot's symbols through `ticker_for_date` before creating the puller.

4. **Top-up download** of only the aliased tickers for the affected dates, written next
   to the existing files:

   ```
   raw/minutes/v1/20250102.ohlcv-1m.dbn.zst       existing, untouched
   raw/minutes/v1/20250102.fix.ohlcv-1m.dbn.zst   new: FI, PARA, MMC, ... for that day
   ```

   Loaders already read every `*.dbn.zst` and take the date from the first 8 characters
   of the file name, so top-up files are picked up automatically. Cost is about 1% of
   the original download. Add this as an option to both pulls (e.g. `--aliases-only`).

5. **Rename old to new when loading** in `load_dbn_bars` (`src/data/raw/dbn_store.py`),
   used by the daily and minutes feature steps. Rename only bars dated on or before
   `until` (New York date): old tickers can be reused later by a different company.
   Stored files keep the original ticker so they remain a faithful copy of the source.

6. **Warn on new renames:** after each pull, print `store.metadata.not_found` in the log,
   so a future rename shows up and a line can be added to the alias table.

## Tests to add

- FI bars before 2025-11-10 and FISV bars after load as one continuous `FISV` series.
- An old ticker dated after `until` is **not** renamed.
- `ticker_for_date` on the change date and the day after.

## Verify after the fix

- Re-read the `not_found` lists: none of the six should appear for dates the alias covers.
- Row counts per day in `features/daily/v1` and `features/minutes/v1` increase by the
  number of recovered stocks.
- Plot FISV close around 2025-11-10: no gap or jump from the rename.

## Interim mitigation

Exclude stocks with fewer than 60 trading days of history from live predictions. This
also covers new listings and new index members.
