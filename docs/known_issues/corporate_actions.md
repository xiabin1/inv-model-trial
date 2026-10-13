# Splits, spin-offs and bad prints in raw prices

**Status:** resolved 2026-09-28. All splits verified against company releases / SEC filings;
spin-offs recorded (WDC, HON x2, DD, CMCSA, FDX, SPGI) plus the HON 1-for-2 reverse split;
bad prints resolved by session bars; BNY ticker reuse handled as a rename
(`ticker_renames.md`).

## Bad prints: resolved

Databento `ohlcv-1d` bars cover the whole UTC day including pre-market and after-hours,
so their open is the first pre-market trade and thin extended-hours prints set many
highs/lows (3,913 bars more than 20% outside open/close; e.g. Gartner "open" 169.81 vs
421.15 at 09:30). Daily features and labels now use regular-session daily bars built from
the 1m data (`src/data/raw/session_bars.py`, built by `python -m src.pull.session_bars`
into `raw/session_daily/v1`). With those, 3 such bars remain and the Gartner/Teledyne
labels are normal (+0.22%, +0.53%).

Remaining caveat: XNAS.BASIC is Nasdaq's feed; for NYSE-listed stocks its first 09:30
trade is close to, but not exactly, the official NYSE opening auction price.

## How splits are handled

- `config/symbols/corporate_actions.yaml` lists splits and reverse splits (and spin-offs
  via `factor`).
- Applied when bars are loaded (`load_dbn_bars`, `load_session_bars`): prices before the
  ex-date are multiplied by the factor, volume divided by it. Raw files are not changed.
- `python3 scripts/detect_corporate_actions.py` lists overnight jumps in the session bars
  that the file does not explain. Run it after new data arrives; verify each candidate
  before adding it (a split-like ratio is not proof: NVR looked like a 1-for-5 reverse
  split in `ohlcv-1d` but was a bad print; MRNA +84% on 2026-08-19 was a real move).

## Spin-offs

A spin-off lowers the parent's price by the value distributed, which the label would read
as a loss. Entries record the inputs, `spinoff: {child, ratio, parent_open, child_open}`, and
the factor is computed as `parent_open / (parent_open + ratio * child_open)` with the
**ex-date opening prices** (`spinoff_factor` in `src/data/raw/corporate_actions.py`).

Why opens: the strategy holds from 15:58 to the next open, and labels and the gap feature
measure previous close -> open. With open prices, the adjusted overnight move into the ex-date
equals what a holder really got (parent + child at the open). Until 2026-10-06 the factors used
ex-date **closes**, which added the two companies' different moves during the ex-date to that
overnight return: CTVA's 9/30 label read +18.8% instead of the real +3.6%. All 9 spin-offs were
switched to opens and the affected days rebuilt in v3, mini_v1, b1542_v1 and b1558_v1 (labels of
D-1, minute features D-1..D+21 sessions, daily features D..D+62 sessions, combined data), and the
live models retrained. Volume is not changed by a spin-off (only splits divide it).

Small spin-offs (Solstice was ~6% of HON, Versant ~5% of CMCSA) stay under the detector's
+-40% threshold, and a spin-off on the same day as a reverse split (HON 2026-06-29) can
cancel out. So after new data, also list symbols that first appear mid-history: a new
listing may be a spin-off whose parent needs a `factor`. Spun-off companies outside the
universe (SNDK in Feb 2025, MBGL, VYLR) need their ex-date open from Databento directly.

## After changing the actions file

Regenerate, in order: daily features, minutes features (only if the split affects the
minutes lookback), labels, combined features.
