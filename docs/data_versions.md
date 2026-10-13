# Data, model and ledger versions

What each version folder holds and how it was made. Paths are relative to
`s3://portfolio-calc-platform-dev/financial/tp1/data/` unless they start with `models/`,
`data/` or `config/` (this repo). Update this file whenever a version is added or retired.

**In use now (live and paper trading):** 15:58 cutoff, Nasdaq Basic (`XNAS.BASIC`), data fix applied:
raw/minutes v1 → features/minutes **v3** + features/daily **v2** → labels **v3** → features/combined **v3**
→ `models/live_classifier.joblib` (variant `weekly_plain_label`).

## Raw data

| Path | Contents | Notes |
|---|---|---|
| `symbols/v1` | Point-in-time S&P 500 + Nasdaq-100 members per day, 2025-01-01 onward | Rebuilt from index change logs (no survivorship bias); current tickers, see `renames` |
| `raw/minutes/v1` | `XNAS.BASIC` 1m bars, one file per session, 2025-01-02 onward | `YYYYMMDD.fix.ohlcv-1m.dbn.zst`: top-up files with the old tickers of renamed symbols (2026-09-28) |
| `raw/session_daily/v1` | Regular-session daily bars built from `raw/minutes/v1` (09:30 open, last regular trade close) | Stored unadjusted; renames and corporate actions are applied when loaded. Rebuilt 2026-09-28 to include the top-up files |
| `raw/daily/v1` | `XNAS.BASIC` ohlcv-1d bars | Not used by the pipeline (whole-UTC-day bars with pre/after-market prints) |
| `raw/minutes_equs_mini/v1` | `EQUS.MINI` 1m bars, 2024-07-01 onward | Mini profile, see below |

Corporate actions, splits, spin-offs and ticker renames are not versioned folders. They live in
`config/symbols/corporate_actions.yaml` and are applied whenever bars are loaded.

## Features and labels

| Version | Cutoff | Bar size | Data fix (renames, spin-offs, verified splits) | Status |
|---|---|---|---|---|
| minutes **v1** | 15:50 | 10m | no | kept for comparison |
| minutes **v2** | 15:58 | 4m | no | kept for comparison |
| minutes **v3** | 15:58 | 4m | yes | **in use** |
| minutes v4 | 15:50 | 10m | yes | stopped part-way on 2026-09-29; files removed, number not reused |
| daily **v1** | (none: previous-session daily bars) | 1d | no | used by combined v1, v2 |
| daily **v2** | (none) | 1d | yes | **in use** (combined v3) |

| Version | Built from | Label | Status |
|---|---|---|---|
| labels **v1** | minutes v1 cutoff price | next 09:30 open / 15:49 close - 1 | comparison |
| labels **v2** | minutes v2 | next open / 15:57 close - 1 | comparison |
| labels **v3** | minutes v3 | next open / 15:57 close - 1 | **in use** |
| combined **v1** | minutes v1 + daily v1 + labels v1 | | comparison |
| combined **v2** | minutes v2 + daily v1 + labels v2 | | comparison |
| combined **v3** | minutes v3 + daily v2 + labels v3 | | **in use** |

The stored label is the raw return; the 0/1 class is made at training time
(`config/train/train_model.yaml`: `model.label` and each walk-forward variant).

## Evaluations

| Path | What | Data |
|---|---|---|
| `evaluation/v1` | Single-split `python -m src.train` (verification 2026-04-28..2026-09-24) | combined v1 (15:50) |
| `evaluation/walk_forward/<variant>` | Walk-forward runs (test 2025-12-01..2026-07-31). The folder has no version: it was written before walk-forward output was versioned | combined v1 (15:50, before data fix) |
| `evaluation/v2`, `evaluation/v2/walk_forward/<variant>` | Single split + walk-forward | combined v2 (15:58, before data fix) |
| `evaluation/v3`, `evaluation/v3/walk_forward/<variant>` | Single split + walk-forward | combined v3 (15:58, data fix) |

Experiment write-up: `docs/experiments/2026-09-28_walk_forward.md` (v1 data).

## Predictions

| Path | What |
|---|---|
| `predictions/v3/YYYY-MM-DD.csv` | Live 15:58 runs (`python -m src.live predict`). 2026-09-25 is a test replay; live streaming has not run yet (no `XNAS.BASIC` live licence) |
| `predictions/v3/late/` | The previous day's 15:58 prediction, made by the 06:30 nightly job from the full-day file |
| `predictions/v3/sim/` | `python -m src.paper simulate` replays (2026-09-21..2026-09-25, weekly_rolling120 sim models) |

## Models (`models/`)

| File | What |
|---|---|
| `live_classifier.joblib` + `.json` | **Live model**: variant `weekly_plain_label` (raw-return top 20% label, all history), combined v3. Refit every Saturday; the `.json` file records the training range |
| `archive/live_classifier_weekly_rolling120.*` | Previous live model (vol-adjusted label, last 120 days), replaced 2026-09-30 |
| `paper_sim_classifier.*` | Last weekly model made by `src.paper simulate` |
| `us_stock_classifier.joblib` | Output of `python -m src.train` (single split), last run on combined v3 |
| `us_stock_classifier_cutoff1550.joblib` | `src.train` model on combined v1 (15:50) |
| `us_stock_classifier_cutoff1558_v3.joblib` | Copy of the `src.train` model on combined v3 |

## Paper ledgers (`data/paper/`)

| File | What |
|---|---|
| `late_trades.csv` | **Current**: top 10 per day of the live model, $100,000/day (`paper.capital_per_day`), filled by the 06:30 nightly job. Rows from 2026-09-28 (top 5 was used 2026-10-01..02; the full top 10 was restored from the archive) |
| `live_trades.csv` | Real-time 15:58 trades (created on the first successful live stream) |
| `sim_trades.csv` | `simulate` run 2026-09-21..2026-09-25 (weekly_rolling120 sim models, top 3) |
| `archive/late_trades_weekly_rolling120.csv` | Late trades of the previous live model (2026-09-28..29) |
| `archive/late_trades_top10_until_20260930.csv` | Top-10 late trades of the current model before the temporary switch to top 5 (restored into `late_trades.csv`) |

## Mini profile (built 2026-10-01/02, being evaluated)

Parallel pipeline: minute features from **EQUS.MINI** (live feed included in the plan), with
daily features, labels and paper prices kept on **XNAS.BASIC**. Selected with `--profile mini`
(`src/live.py` PROFILES). Configs: `config/feature/*_mini.yaml`, `config/train/train_model_mini.yaml`.

| Path | Contents |
|---|---|
| `raw/minutes_equs_mini/v1` | EQUS.MINI 1m bars, 2024-07-01 onward (config `config/data/minutes_equs_mini_config.yaml`) |
| `raw/minutes/v1`, 2024-07..2024-12 | XNAS.BASIC 1m bars added for the longer history (plus `symbols/v1` 2024-06..12, session bars, 2024 splits and the J/Amentum spin-off in `corporate_actions.yaml`) |
| `features/daily/v2`, 2024-10-01..2025-06-04 | Earlier dates added to the existing daily version (same definition) |
| `features/minutes/mini_v1` | 15:58 cutoff, 4m bars, from EQUS.MINI; `ret_day`/`ret_gap` use `open_5m` (first-5-minute VWAP) instead of the first trade. From 2024-08-01 |
| `labels/mini_v1` | next XNAS.BASIC 09:30 open / last XNAS.BASIC 1m close before 15:58 - 1 (`cutoff_raw` option) |
| `features/combined/mini_v1` | minutes mini_v1 + daily v2 + labels mini_v1, from 2024-10-01 |
| `evaluation/mini_v1/...` | Walk-forward runs of the mini profile |
| `predictions/mini_v1/` | Mini predictions (`late/` subfolder for the nightly after-the-fact runs) |
| `models/live_classifier_mini.*` | Mini live model (`weekly_plain_label`) |
| `data/paper/late_mini_trades.csv`, `live_mini_trades.csv` | Mini paper ledgers |

## b1542 profile (live trading model from 2026-10-05) and the cutoff experiment

Experiment 2026-10-04 (same XNAS.BASIC source, history and labels; only the minute-feature cutoff
differs) found the 15:42 cutoff costs little versus 15:58, and b1542 is the best model that runs live
on the plan: same-day XNAS.BASIC is 15 minutes delayed, so the bars before 15:42 are ready ~15:57.

| Path | Contents |
|---|---|
| `features/minutes/b1542_v1` | XNAS.BASIC, bars 09:30..15:42 (cutoff_before_close 18), 4m bars, from 2024-08-01 |
| `features/minutes/b1558_v1` | Same with bars to 15:58: the reference (= v3's recipe with the longer history) |
| `features/combined/b1542_v1`, `b1558_v1` | + daily v2 + `labels/mini_v1`, from 2024-10-01 |
| `labels/mini_v1` | Shared by mini, b1542, b1558: XNAS.BASIC 15:57 close -> next XNAS.BASIC open (the name is historical; no EQUS.MINI data in it) |
| `evaluation/b1542_v1/...`, `evaluation/b1558_v1/...` | Walk-forward (weekly_plain_label; test 2025-12..2026-07 and 2025-06..2026-07) |
| `predictions/b1542_v1/` | b1542 live predictions; `late/` = nightly after-the-fact runs |
| `models/live_classifier_b1542.*` | b1542 live model (weekly_plain_label), refit every Saturday |
| `data/paper/late_b1542_trades.csv`, `live_b1542_trades.csv` | b1542 paper ledgers (backfilled from 2026-09-28 with models trained only on earlier labels) |
| `data/paper/ibkr_trades.csv` | IBKR paper account orders/fills (`config/broker/ibkr_config.yaml`, profile b1542) |

Configs: `config/feature/*_b1542.yaml`, `*_b1558.yaml`, `config/train/train_model_b1542.yaml`, `train_model_b1558.yaml`.
The basic (v3) live 15:56 job was removed (needs the XNAS.BASIC live licence); its nightly still builds the shared data.

## Planned

Nothing pending.
