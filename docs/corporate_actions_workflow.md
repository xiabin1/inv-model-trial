# Corporate actions: detection and approval workflow

How splits, reverse splits and spin-offs are found, checked and applied. Raw data is never
changed: events are recorded in `config/symbols/corporate_actions.yaml` and applied whenever
bars are loaded (`src/data/raw/corporate_actions.py`). Background and history:
`docs/known_issues/corporate_actions.md`, `docs/known_issues/unrecorded_corporate_actions.md`.

## Why it matters

An unrecorded event looks like a fake crash or jump: Corteva's spin-off of Vylor (2026-10-01)
looked like -81% overnight. Labels, gap features and 60-day indicators are then wrong, and
the stock can be picked for the wrong reason.

## Adjustment rule

Prices **before** the ex-date of the affected stock are multiplied by the action's factor
(open, high, low, close, VWAP, in daily and 1-minute bars). Bars on/after the ex-date and other
stocks are unchanged.

| Event | Entry | Factor | Volume |
|---|---|---|---|
| Split / reverse split | `split: 10` (new shares per old; 1-for-5 reverse = 0.2) | 1 / split | divided by the factor |
| Spin-off | `spinoff: {child, ratio, parent_open, child_open}` | parent_open / (parent_open + ratio x child_open), with **ex-date opening prices** | unchanged |
| Other | `factor: x` | x | unchanged |

Spin-offs use the ex-date **opens** because the strategy holds from 15:58 to the next open: the
adjusted overnight move into the ex-date then equals what a holder really got (parent + child
at the open). With closes, CTVA's 9/30 label read +18.8% instead of the real +3.6%.

## Detectors

Both compare **open / previous close** after the recorded actions are applied, with the same rule:

```
ratio = open / previous close
flagged if  ratio is within 8% of a split ratio (1/2, 1/3, 1/4, 1/5, 1/10, ... or 2, 3, ...)
       or  |ratio - 1| > 40%            (live.gap_guard_max_move)
```

| | Gap guard | Nightly check |
|---|---|---|
| Code | `gap_guard` in `src/live.py` | `check_corporate_actions` in `src/live.py` |
| Runs | ex-date D, ~15:57, inside `live predict` (b1542) | D+1 at 06:30, in the basic `live nightly` |
| Gap checked | D-1 -> D (the `min_ret_gap` feature the model uses) | D-1 -> D, from the session bars |
| Previous close | D-1's last price before the cutoff (15:41 for b1542, 15:57 for 15:58 models) | D-1's official regular-session close |
| Open | D's 09:30 first trade (mini: first-5-minute VWAP) | D's regular-session open |
| Purpose | protect D's prediction: exclude the stock immediately | backup (holiday, failed run) and morning alert |

A flagged stock gets a **pending** entry (once per symbol and date; whichever detector is first),
a line in `logs/corporate_action_alerts.log`, and a Telegram message. The split guess is only a
starting point: the detectors see prices only and cannot tell a split from a spin-off or news.

### Examples (real events)

| Case | Prev close -> open | Ratio | Flagged | Entry guess | Right decision |
|---|---|---|---|---|---|
| NFLX 10-for-1 (2025-11-17) | 1115 -> 112 | 0.100 | yes (~1/10) | `split: 10` | Verified |
| CTVA spin-off (2026-10-01) | 77.67 -> 14.44 | 0.186 | yes (near 1/5, -81%) | `split: 5` (wrong) | edit to `spinoff: {child: VYLR, ratio: 1}`, then Verified |
| MRNA news (2026-08-19) | 62.97 -> 116.05 | 1.843 | yes (+84%, near 2) | `split: 0.5` (wrong) | Reject (real move) |
| AMCR 1-for-5 reverse (2026-01-15) | ~5x | ~5.0 | yes | `split: 0.2` | Verified |
| HON / Solstice spin-off (2025-10-30) | 212.84 -> 202.14 | 0.950 | **no** (-5%) | - | missed: needs advance sources (OCC memos) |
| STX normal big move (2026-10-02) | +13% | 1.13 | no | - | - |

**Spin-off filing + gap check** (basic nightly, with the nightly check): a drop of 10% or more that the
rule above does not flag, in a company with an 8-K mentioning a spin-off or distribution ratio in the
last 120 days, is proposed the same way (about 1 alert a month; `spinoff_gap_check` in
`src/corporate_action_scan.py`). Training also leaves out overnight moves beyond +-40%
(`trainable` in `src/walk_forward.py`), so a missed event cannot enter the weekly refit.

Known limits: small spin-offs (a few % of the parent) are invisible to price gaps, and an event is
only seen on its ex-date. Advance sources (OCC information memos, below) address both.

## Approval workflow (`src/corporate_action_flow.py`)

| Entry state | Meaning | Prices | Stock in picks |
|---|---|---|---|
| `verified: false` | pending (proposed) | not adjusted | **excluded** |
| `verified: true`, not yet rebuilt | approved, rebuild running | adjusted for new computations | **excluded** |
| `verified: true`, rebuilt | applied | adjusted everywhere | included |
| `rejected: true` | false alarm (e.g. real news); kept so it is not proposed again | not adjusted | included |

```
detector -> propose(): pending entry + Telegram (Verify / Reject buttons) -> stock excluded
you      -> check the company's announcement; tap a button (or edit the file)
             unknown / spin-off: first add spinoff: {child: TICKER, ratio: N} to the entry
check()  -> every 15 min 09:00-17:45 and at 07:15 (never 15:54-16:02):
             read taps (only from TELEGRAM_CHAT_ID) -> set the flag in the file
             verified: fill spin-off opens (session bars / Databento), rebuild the affected days in
             v3, mini_v1, b1542_v1, b1558_v1, retrain basic/mini/b1542 -> Telegram confirmation
             rejected: record it -> Telegram confirmation
ack()    -> every minute: records a tap right away, answers it and says when it will be applied
remind() -> 08:30 Mon-Fri: Telegram list of pending entries
```

Rebuild windows for ex-date D: labels of D-1; minute features D-1..D+21 sessions; daily features
D..D+62 sessions; combined data D-1..D+62 sessions. State: `data/state/applied_actions.json`
(applied/rejected entries), `data/state/telegram_updates.json` (Telegram offset).

Commands:

```
python -m src.corporate_action_flow pending                      # waiting entries, excluded stocks
python -m src.corporate_action_flow check                        # taps + apply (cron)
python -m src.corporate_action_flow remind                       # reminder (cron)
python -m src.corporate_action_flow propose SYMBOL DATE --gap R   # add a pending entry by hand
python -m src.corporate_action_flow apply SYMBOL DATE             # rebuild one event by hand
```

## Advance detection (`src/corporate_action_scan.py`)

Runs at 07:30 and 18:30 Mon-Fri (`python -m src.corporate_action_scan scan`), remembering what it
reported in `data/state/corporate_action_scan.json`.

| Source | Covers | Result |
|---|---|---|
| Nasdaq Trader corporate-action alerts (RSS + alert pages, "ECA") | Nasdaq-listed stocks | for our universe: a **pending entry with the terms filled in** (`propose_known`) and a Telegram message with Verify / Reject, usually days before the ex-date |
| SEC EDGAR full-text search (8-K, Form 10: "stock split", "reverse stock split", "spin-off", "distribution ratio") | all listed companies, often weeks ahead | Telegram **heads-up digest** only (filings rarely state a clean ex-date). Needs `SEC_USER_AGENT` in `.env` (SEC asks automated clients for a contact); skipped otherwise |

Parsing (tested on real alerts): spin-offs read "EX Date: January 5, 2026" and "One share of ... (VSNT)
for every twenty five shares ... held" -> `spinoff: {child: VSNT, ratio: 0.04}`; splits read
"one-for-ten (1-10) reverse split ... effective on Thursday, October 8, 2026" or "split-adjusted basis
on ..." -> `split: 0.1`. Alerts without an ex-date or terms yet are looked at again on later scans.

Announced events wait for their ex-date: a verified entry dated in the future does not change prices
(`load_actions` ignores it until the ex-date), the stock is still picked before the ex-date and excluded
from it, and the rebuild runs once the ex-date's data exists (spin-off opens are filled then).

Look-back audit: `python -m src.corporate_action_scan audit --start 2024-07-01` lists EDGAR filings for
our stocks that no recorded event explains (within 60 days), in `logs/corporate_action_audit.csv`.

Full audit, 2026-10-07 (per company, 2024-07..2026-10, 438 companies, ~1,000 filings): keyword hits alone are
mostly boilerplate, so they were crossed with prices. (1) The detector rule over all 568 stored sessions
(actions applied) flags only real news moves (BE 2024-11-15, ECHO 2025-06-16 / 2025-08-26, FISV 2025-10-29,
MRNA 2026-08-19): no missed split. (2) For companies with spin-off / distribution-ratio filings, the largest
down-gap within 120 days, checked against the filings, found three missed spin-offs, proposed for approval:
FTV 2025-06-30 (RAL, 1 per 3, -26.8%), BDX 2026-02-10 (WAT, 0.135 per share, Reverse Morris Trust, -20.4%),
APTV 2026-04-01 (VGNT, 1 per 3, -13.4%). The other down-gaps were earnings or news days.

Not used: OCC information memos (the site blocks automated access with a bot check).
