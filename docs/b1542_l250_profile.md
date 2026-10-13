# b1542_l250: b1542 trained on a wider universe, max 4 picks per sector

Added 2026-10-08 as a **shadow** of b1542 (paper trading only). IBKR keeps trading `b1542` until
`strategy.profile` in `config/broker/ibkr_config.yaml` is changed to `b1542_l250` (and `dry_run`
stays the separate switch for real orders).

## What it is

| | b1542 (current IBKR model) | b1542_l250 |
|---|---|---|
| Features | XNAS.BASIC bars up to 15:42 + daily features | same definitions |
| Training universe | S&P 500 + Nasdaq-100 (~519) | the same + liquid US stocks (~100-160): common stock or ADR, price >= $5, median 20-day dollar volume >= $250M (leave below $200M, 15 of 20 days >= $125M first) |
| Picks | top 10 of the current stocks | top 10 of the current stocks, **at most 4 per GICS sector** |
| Retrain | Saturday 09:30 | Saturday 09:45 |

Why: walk-forward tests 2025-06..2026-10 (scripts/experiments/, results in the 2026-10-08 session):
whole 336 days Sharpe 1.93 vs 1.52, max drawdown -10.9% vs -16.3%, 32 vs 40 days below -2%; on the
untouched holdout 2026-08-01..10-06 +0.17%/day vs +0.08%/day. Training on the wider universe gave
+0.07%/day (97% bootstrap chance > 0); the sector cap mainly lowers the risk.

## Daily flow

| Time (ET) | Job | What |
|---|---|---|
| 07:00 | `live nightly --profile b1542` | as before; its late prediction also scores the shadow -> ledger `late_b1542_l250` |
| 07:20 | `live nightly --profile b1542_l250` | EQUS.SUMMARY daily bars of all symbols (`raw/daily_all/v1`), next session's lists (`symbols/l250_v1`, `symbols/l250_v1_pull`), minute download for the lists, session bars, daily and minute features, labels, combined data (`features/combined/l250_b1542_v1`), unrecorded-split check on the extra stocks |
| 15:56 | `live predict --profile b1542 --trade` | b1542 picks as before; then the shadow scores the same rows (~0.1 s) -> ledger `live_b1542_l250`; IBKR buys the picks of whichever profile `ibkr_config.yaml` names |
| 09:48 | paper settle | includes `late_b1542_l250`, `live_b1542_l250` |

A failing shadow never stops b1542's picks or orders (it is wrapped and logged).

## Files

- `src/symbols/liquidity_universe.py`: daily bars of all symbols, security types, the two lists (`update --date D`).
- `config/symbols/sectors.csv`: GICS sector and sub-industry (S&P 500 list; 62 others by hand). Stocks
  missing from it are not capped and are named in the prediction log; add new index members there.
- Configs: `config/data/minutes_config_l250.yaml`, `session_bars_config_l250.yaml`,
  `config/feature/*_l250.yaml`, `config/train/train_model_b1542_l250.yaml` (`live.sector_cap: 4`).
- Data versions keep the experiment's history: raw/minutes and session_daily `l50_v1`, features
  `l50_v2` / `l50_b1542_v1`, labels `l50_mini_v1`, combined `l250_b1542_v1`.
- Corporate-action rebuilds (`src/corporate_action_flow.py`) include the `_l250` configs; splits of the
  extra stocks are not recorded yet (IBKR 4-for-1, 2025-06-18, is left out of training by the +-40% filter).

## Pick rules for new listings (all profiles, 2026-10-08)

A stock is scored but **not picked** when
- less than 90% of its features are present (`live.min_feature_share`), e.g. during its first two days in
  the index, when its daily features do not exist yet; or
- it has traded for fewer than 20 sessions (`live.min_listing_days`), counted from its first day in the
  all-market daily bars (`raw/daily_all/v1/first_seen.parquet`, updated nightly by the b1542_l250 run), so a
  long-listed stock that just joined the index is not held back, and a renamed ticker keeps its history.
The skipped stocks are named in the prediction log. Example: VYLR (Corteva spin-off, listed 2026-10-01).

Sectors: `python -m src.symbols.sectors refresh` (Saturday 08:30) updates `config/symbols/sectors.csv` from
the S&P 500 GICS list and sends a Telegram message if a current index member has no sector.

## mini_l250 (added 2026-10-10)

The same idea on top of **mini**: EQUS.MINI minute features with data to **15:58**, trained on the same wider
universe, picks from the current stocks, **at most 3 per sector** (`config/train/train_model_mini_l250.yaml`).
A shadow of mini: at ~15:58 it scores the rows mini's live run has just built; paper ledgers
`live_mini_l250` / `late_mini_l250`; Telegram picks and sell messages (`compare` in config/notify/telegram.yaml).

Test (scripts/experiments/mini_l250.py, 336 days 2025-06..2026-10-06): Sharpe 2.29 (best of all models),
total +118%, max drawdown -12.3%, 27 days below -2%; Sharpe 2.16 / 2.24 / 2.71 in 2025-06..11 / 2025-12..2026-07 /
holdout. The wider training fixed mini's weak 2025 period (0.12%/day -> 0.23%/day).

| Time (ET) | Job |
|---|---|
| 07:20 Tue-Sat | `nightly --profile b1542_l250`, **then** `nightly --profile mini_l250` (same cron line: mini_l250 needs b1542_l250's lists, daily features and labels): EQUS.MINI download for `symbols/l250_v1_pull`, mini-style minute features (`l250_mini_v1`), labels, combined (`features/combined/l250_mini_v1`) |
| 15:56 | `predict --profile mini`: mini's picks, then mini_l250's (shadow, ~0.2 s) |
| 09:48 | paper settle includes `late_mini_l250`, `live_mini_l250` |
| Sat 10:00 | `retrain --profile mini_l250` |

Configs: `config/data/minutes_equs_mini_config_l250.yaml`, `config/feature/minutes_feature_config_mini_l250.yaml`,
`config/feature/combined_feature_config_mini_l250.yaml`; raw data `raw/minutes_equs_mini/l250_v1`. Corporate-action
rebuilds include the `_mini_l250` configs.
