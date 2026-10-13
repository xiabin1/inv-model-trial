# Walk-forward test of retraining schedules and labels (2026-09-28)

## Question

If the model is retrained on a schedule, as in live use, does it pick stocks that beat
the market overnight after costs? Which schedule, training window and label work best?

## Setup

- **Data:** combined features (169,251 rows, 329 trading days, 328 features), regular-session
  daily bars, split-adjusted (13 corporate actions), labels = next 09:30 open / 15:49 price - 1.
- **Walk-forward:** before each test period, train a fresh LightGBM (same parameters every time,
  `config/train/train_model.yaml`) on the labels known by then. For a period starting on day D,
  the newest usable label is from two sessions before D (Friday's label needs Monday's open).
- **Test period:** 2025-12-01 .. 2026-07-31, 166 trading days, all out of sample.
- **2026-08-01 onward: not used by the walk-forward.** Note: it is *not* a clean holdout. The
  single-split runs (`python -m src.train`, verification 2026-04-28 .. 2026-09-24, and the
  plain vs volatility-adjusted comparison on that period) already included Aug-Sep 2026 and
  influenced choices made here. A clean final check needs new data (paper trading).
- **Strategy:** each day buy the N highest-scored stocks equally at 15:49, sell at the next
  09:30 open; 0.1% cost per day (round trip).
- **Baselines:** all stocks (overnight drift), and the most volatile 20% of stocks by trailing
  overnight volatility (what a model that only finds volatile stocks would earn).
- **Runtime:** about 20 minutes for all five variants (6-10 s per weekly model).

| Variant | Retrain | Training window | Label |
|---|---|---|---|
| weekly_expanding | weekly | all history | top 20% by excess return / own volatility |
| monthly_expanding | monthly | all history | same |
| never | once, before 2025-12-01 | all history | same |
| weekly_rolling120 | weekly | last 120 trading days | same |
| weekly_plain_label | weekly | all history | top 20% by raw return |

## Results

### Ranking quality (does the score order stocks correctly?)

| Variant | Rank IC | t-stat | ICIR / yr | Positive months | Top-decile excess / day | Decile spread / day |
|---|---|---|---|---|---|---|
| weekly_expanding | 0.0198 | **2.26** | 2.79 | 88% | +0.094% | 0.072% |
| monthly_expanding | 0.0221 | **2.60** | 3.20 | 100% | +0.086% | 0.062% |
| never | 0.0106 | 1.30 | 1.60 | 75% | +0.055% | 0.010% |
| weekly_rolling120 | 0.0236 | **2.76** | 3.40 | 100% | +0.114% | 0.094% |
| weekly_plain_label | 0.0275 | 1.51 | 1.86 | 62% | +0.157% | 0.193% |

### Strategy after 0.1% daily cost (annual return, Sharpe, t-stat of daily return)

| Variant | Top 3 | Top 5 | Top 10 | Top 30 | Top 50 |
|---|---|---|---|---|---|
| weekly_expanding | -17.8% | -24.5% | -13.1% (-0.49, t -0.40) | -0.7% (0.04, t 0.03) | 2.1% (0.22, t 0.18) |
| monthly_expanding | -15.8% | -0.6% | 7.8% (0.44, t 0.36) | 9.6% (0.64, t 0.52) | 1.5% (0.18, t 0.14) |
| never | 86.0% (t 1.89) | 37.7% | -2.5% (-0.03, t -0.02) | -6.0% (-0.37, t -0.30) | -6.9% (-0.53, t -0.43) |
| weekly_rolling120 | -14.6% | -10.5% | 17.2% (0.81, t 0.66) | **22.8% (1.34, t 1.08)** | 8.1% (0.65, t 0.53) |
| weekly_plain_label | 67.0% (t 1.09) | 58.1% (t 1.05) | **79.8% (1.73, t 1.40)** | 33.7% (1.13, t 0.92) | 16.7% (0.77, t 0.62) |

Edge over the most-volatile-20% baseline (t-stat): best is weekly_plain_label top 10
(+0.25%/day, **t 1.97**), then weekly_rolling120 top 30 (+0.08%/day, t 1.29).
All stocks, same 0.1% cost: negative in 6 of 8 months (overnight drift < cost).

### Top 10 by month (sum of daily returns after cost)

| Month | weekly_exp | monthly_exp | never | rolling120 | plain_label | all stocks |
|---|---|---|---|---|---|---|
| 2025-12 | -1.3% | +0.1% | +0.1% | +0.8% | +4.7% | -1.6% |
| 2026-01 | +1.5% | -0.6% | +1.9% | +4.7% | +8.2% | -1.9% |
| 2026-02 | -7.7% | -5.3% | +0.7% | -1.5% | -5.9% | -3.4% |
| 2026-03 | -8.5% | -4.5% | -6.8% | -5.7% | -4.0% | -5.3% |
| 2026-04 | -1.5% | +9.2% | +11.8% | +2.5% | +17.0% | +0.9% |
| 2026-05 | +3.0% | +3.0% | +5.1% | +2.5% | +9.4% | -1.8% |
| 2026-06 | +9.7% | +3.0% | +0.5% | +6.8% | +9.5% | +0.7% |
| 2026-07 | -2.6% | +1.8% | -13.8% | +2.1% | +4.6% | +1.0% |

## Interpretation

1. **Retraining matters.** `never` has half the ranking signal of the retrained variants and
   is not significant. Retraining weekly or monthly keeps the signal alive.
2. **Recent data helps.** `weekly_rolling120` has the best and most consistent ranking
   (t 2.76, positive every month) and the best vol-adjusted trading result.
3. **The ranking is real but small.** With the volatility-adjusted label, only the top decile
   clearly beats the day's median (~0.1%/day), about the size of the 0.1% cost.
4. **Plain label trades best, but depends on big moves.** Its picks move 2.35% overnight vs
   0.89% for all stocks; on 7.8% of days its best pick gains over 10%. Without its 5 best
   days the top-10 mean drops from 0.262% to 0.051% per day. Its ranking is less steady
   (positive in 62% of months). It still beats the volatile-stock baseline (t 1.97), so it is
   not only picking volatility.
5. **Top 3 is not reliably better.** Negative in 3 of 5 variants, worst days near -9.5%,
   negative median days. The +86% for `never` comes from the weakest-ranking model: luck.
6. **Nothing is proven yet.** No strategy t-stat reaches 2. With 5 variants x 5 portfolio
   sizes tried, one result looking good by chance is expected.
7. **February-March 2026 were losing months for every variant**, which suggests a market
   regime the features do not handle.

## Next steps

- Test **weekly_rolling120 with the plain label** (combines the two best parts).
- Pick one setup and confirm it with **paper trading on new days**. Aug-Sep 2026 can be run
  as a semi-clean check (`python -m src.walk_forward --variant <name> --include-holdout`), but
  it was already seen in the single-split runs, so a good result there counts for less.
- Look at February-March 2026 (market moves, which stocks lost).
- Consider an earnings-date flag: many of the largest moves are news.
- Check real trading costs; 0.1% per round trip is conservative for liquid large caps.

## Files

- Code: `src/walk_forward.py`, `score_report` in `src/models/evaluate.py`;
  variants in `config/train/train_model.yaml` (`walk_forward`).
- Results per variant: `s3://portfolio-calc-platform-dev/financial/tp1/data/evaluation/walk_forward/<variant>/`
  (`summary.json`, `periods.csv` per retrain period, `daily_ic.csv`, `deciles.csv`,
  `predictions.parquet`).

## Follow-up: shorter window and daily retraining

Same test period (2025-12-01 .. 2026-07-31, 166 days), volatility-adjusted label, 0.1% cost.

| Variant | Rank IC (t) | Top 3 | Top 5 | Top 10 | Top 30 | Top 50 |
|---|---|---|---|---|---|---|
| weekly_rolling30 (35 models) | 0.0194 (1.93) | | | -11.9% (t -0.48) | -6.4% (t -0.35) | -5.0% (t -0.33) |
| weekly_rolling120 (35 models) | 0.0236 (2.76) | -15% (t -0.24) | -11% (t -0.23) | +17% (t 0.66) | **+23% (t 1.08)** | +8% (t 0.53) |
| daily_rolling120 (166 models) | 0.0238 (2.81) | -8% (t -0.07) | +2% (t 0.18) | **+25% (t 0.89)** | +11% (t 0.62) | +5% (t 0.39) |

- **30 trading days is too little history:** weaker ranking, every portfolio size loses after cost.
- **Daily vs weekly retraining (120-day window): no real difference.** Same ranking quality
  (Rank IC 0.0238 vs 0.0236), scores correlate 0.84 and the two share 5.8 of their top 10
  picks per day. Daily is a bit better for top 10, weekly for top 30; both within noise.
  Monthly results move together, including the losing February-March 2026.
- A 30-day look at daily retraining alone (`daily_rolling120_last30`, 2026-06-18 .. 07-31) was
  too short to judge (all t-stats below 0.7).
- **Conclusion:** retraining on the last 120 trading days is what matters; weekly is enough.
  Daily retraining is fine if it is operationally easy, but it does not add measurable value.

## Follow-up: original label (return > 1%) with weekly retraining, 120-day window

Variant `weekly_rolling120_threshold`; checks by `scripts/analyze_walk_forward.py`.
Same 166 days, 0.1% cost, compared with `weekly_rolling120` (volatility-adjusted label).

| | Original label | Volatility-adjusted label |
|---|---|---|
| Rank IC (t, positive months) | 0.0239 (1.38, 75%) | 0.0236 (2.76, 100%) |
| Top-decile excess / day | +0.165% | +0.114% |
| Top 3 / 5 | +206% (t 2.22) / +106% (t 1.69) | -15% / -11% |
| Top 10 | +48.5%, Sharpe 1.39, maxDD -18.1%, t 1.13 | +17.2%, Sharpe 0.81, maxDD -10.7%, t 0.66 |
| Top 30 | +26.9%, Sharpe 1.07, maxDD -16.2%, t 0.87 | +22.8%, Sharpe 1.34, maxDD -9.8%, t 1.08 |
| Top 50 | +22.1%, Sharpe 1.05, maxDD -14.5%, t 0.85 (vs high-vol t 2.02) | +8.1%, t 0.53 |
| Positive months (top 10 / 30) | 5/8, 6/8 | 6/8, 6/8 |

Month by month (top 10): Dec +7.2%, Jan +9.0%, Feb -4.9%, Mar -4.9%, Apr +9.5%, May +10.0%,
Jun +4.6%, Jul -1.1%. February-March lost about 10%, as in every variant.

**Where the profit comes from:**
- **Leveraged market exposure:** on up-market nights top 10 gained +1.16% (market +0.37%), on
  down nights it lost -0.67% (market -0.34%); the picks move about 2-3x the market both ways.
- **A few days:** the best 5 of 166 days are 91% (top 10), 110% (top 30) and 124% (top 50) of
  the total; without them the daily mean is +0.017%, -0.011% and -0.022%.
- **One theme:** 191 different stocks in the top 10, but the 10 most frequent are 40% of picks
  (BE 99 days, MU 96, SNDK 89, WDC 87, STX 58, MRNA 52), mostly memory/storage/AI hardware.
- Picks' trailing overnight volatility 2.49% vs 0.94% for all stocks.

**Cost sensitivity (annual):**

| | Break-even | 0.05% | 0.10% | 0.15% | 0.20% |
|---|---|---|---|---|---|
| Top 10 | 0.28%/day | +68% | +49% | +31% | +15% |
| Top 30 | 0.21%/day | +44% | +27% | +12% | -1% |
| Top 50 | 0.19%/day | +38% | +22% | +8% | -5% |

**Conclusion:** the original label's higher returns come mainly from high beta, a few big days
and one sector theme, not from robust stock selection; it would likely lose 2-3x the market in
a falling market. The volatility-adjusted label has a smaller but steadier selection signal and
is the better candidate for paper trading. Ideas to test from saved predictions: inverse-
volatility weighting, limits on repeated names/sectors, and a beta-adjusted (market-neutral)
comparison.

## Follow-up: not trading before weekends

Question: avoid holding over the weekend (weekend news risk) by not buying on Fridays.
From saved predictions (no retraining), `scripts/analyze_walk_forward.py` section 5.
Weekly retraining, 120-day window, 0.1% cost; skipped days earn 0 and cost nothing.

Average return per trade by buy day (top 10; 32-35 trades each):

| Label | Mon | Tue | Wed | Thu | Fri |
|---|---|---|---|---|---|
| Volatility-adjusted | -0.59% | +0.18% | +0.41% | -0.06% | +0.39% |
| Original (> 1%) | -0.32% | +0.58% | +0.24% | -0.05% | +0.42% |

Annual return after cost (max drawdown):

| Label, portfolio | Every day | Skip Fridays | Skip days before weekends/holidays |
|---|---|---|---|
| Vol-adjusted, top 10 | +17.2% (-10.7%) | -2.8% (-12.9%) | -7.8% (-14.2%) |
| Vol-adjusted, top 30 | +22.8% (-9.8%) | +7.5% (-9.0%) | +4.9% (-9.0%) |
| Vol-adjusted, top 50 | +8.1% (-11.1%) | -0.3% (-9.4%) | -1.8% (-9.4%) |
| Original, top 10 | +48.5% (-18.1%) | +22.2% (-15.4%) | +9.2% (-16.3%) |
| Original, top 30 | +26.9% (-16.2%) | +6.4% (-15.3%) | -2.3% (-15.9%) |
| Original, top 50 | +22.1% (-14.5%) | +6.5% (-13.8%) | -0.2% (-14.2%) |

- In this period Friday-to-Monday trades were among the best; skipping them removed much of
  the return and barely reduced drawdowns. Only 32 weekends, so noisy.
- Do not pick trading days from the weekday table (e.g. "skip Mondays"): with ~32 trades per
  weekday, one weak weekday is expected by chance.
- Gentler options if weekend risk is a concern: smaller Friday positions, or hedging the
  market (e.g. short an index ETF) on Fridays. Not tested yet.
