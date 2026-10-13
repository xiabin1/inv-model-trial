# Unrecorded splits and spin-offs: remaining protections

**Status:** 2 and 3 done (2026-10-08); 1 open, waiting for a decision (see below).

## Background

Raw Databento prices are unadjusted. Splits and spin-offs are corrected when bars are loaded,
using `config/symbols/corporate_actions.yaml`. An event that is **not** in that file shows up
as a fake overnight crash or jump. Corteva (CTVA) spun off Vylor (VYLR) on 2026-10-01: CTVA went
77.67 -> 14.44 overnight, which the data read as -81%. It went unnoticed for a weekend: b1542
picked CTVA, and the 9/30 label was a fake -81% (fixed 2026-10-04, factor 0.15642).

Timeline for an event with ex-date D (first trading day at the new price):

| When | What is affected |
|---|---|
| D-1 15:58 prediction | nothing (data still normal) |
| D-1 -> D overnight, if bought on D-1 | the trade's result (paper ledger) and the share count held at IBKR |
| D 15:58 prediction | the features (gap reads as a crash or jump) |
| D+1 onward | the indicators: 60-day windows stay distorted for weeks until the event is recorded |

## Done

- **Same-day gap guard** (`src/live.py`, `gap_guard`; test in `tests/test_corporate_actions.py`).
  At prediction time, stocks whose overnight gap (`min_ret_gap`, after recorded actions) is
  split-like (within 8% of 1/2, 1/3, ... or 2, 3, ...) or beyond +-40% (`live.gap_guard_max_move`)
  are excluded from the day's picks and written to `logs/corporate_action_alerts.log`.
- **Nightly check** (`check_corporate_actions` in `src/live.py`, basic nightly at 06:30): writes the
  same alert for the previous session's gaps.
- **Spin-off volume fix** (`src/data/raw/corporate_actions.py`): a `factor` (spin-off) no longer
  changes volume; only splits do.

- **Check-and-approve workflow** (`src/corporate_action_flow.py`, 2026-10-07). The gap guard and
  the nightly check now *propose* a pending entry (`verified: false`) in
  `config/symbols/corporate_actions.yaml` and send a Telegram message with Verified / Reject
  buttons. Pending entries are **not applied** (`load_actions` only applies `verified: true`), and
  the stock is **excluded from the picks** while pending or verified-but-not-yet-rebuilt. A check
  every 15 minutes (07:15 and 09:00-17:45, never 15:54-16:02) reads the taps, fills spin-off
  opening prices, rebuilds the affected days in all four versions, retrains the live models and
  confirms in Telegram. Daily 08:30 reminder for pending entries. Manual rebuild:
  `python -m src.corporate_action_flow apply SYMBOL DATE`.
- **Spin-off factors use ex-date opens** (2026-10-06), see `corporate_actions.md`.

- **Spin-off filing + gap check** (to do item 2, 2026-10-08; `spinoff_gap_check` in
  `src/corporate_action_scan.py`, called by the basic nightly's `check_corporate_actions`). A stock
  with an overnight drop of 10% or more (after recorded actions) whose company filed an 8-K
  mentioning a spin-off or distribution ratio in the last 120 days gets a pending entry (excluded
  from the picks) and a Telegram message with the filing link. On 2024-07..2026-10 it would have
  raised about 1 alert a month and caught every spin-off with a gap of 10%+ (APTV, BDX, CTVA, DD,
  FDX, FTV). Smaller spin-offs (HON 2025-10-30, -5%) remain for the Nasdaq alerts.
- **Training label filter** (to do item 3, 2026-10-08; `trainable` in `src/walk_forward.py`, used by
  the walk-forward and `live.retrain`). Training rows with an overnight move beyond +-40%
  (`model.max_train_move`, default 0.40) are left out; test days keep every row. Today this drops 5
  of 258,832 b1542 rows, all real news moves (BE, ECHO x2, FISV, MRNA).

## To do

1. **Held through a split or spin-off** (affects real IBKR orders once `dry_run: false`). Not
   implemented; the user is considering it. Points agreed so far (2026-10-08):
   - The problem: bought at 15:58, event overnight. IBKR then holds e.g. twice the shares (2:1
     split), or also shares of the spun-off company; the 09:00 sell is capped at the shares bought,
     so the rest stays. The paper ledger records a fake loss or gain for an unrecorded event.
   - **Never "sell everything IBKR holds"**: the account may also hold shares the user bought by hand.
     Sell only what the strategy should hold after the event:
     - event recorded in corporate_actions.yaml: split -> bought x split (capped at IBKR's position);
       spin-off -> the parent shares bought plus floor(bought x ratio) shares of the child;
     - IBKR holds more than bought but no event is recorded: sell only the shares bought and send a
       Telegram alert (split? spin-off? own shares?); no guessing.
   - Paper ledger: a trade whose result is split-like or beyond +-40% without a decided event is
     marked "check event" instead of closed, and settled after Verify (adjusted prices) or Reject.
   - Preferred setup: a separate IBKR account only for the strategy, so a mismatch can only mean
     a corporate action or a failed order. The code then needs `account:` in
     config/broker/ibkr_config.yaml, used for every order, position and fill (today no account is named).

Optional later: an advance check against a public split/spin-off calendar, to record events before
their ex-date.

## Until then

When the gap guard or the nightly check writes an alert, verify the event (company release /
SEC 8-K), add it to `config/symbols/corporate_actions.yaml`, and rebuild the affected days as in
the CTVA fix: daily and minute features from D, labels for D-1, combined data, then retrain.
